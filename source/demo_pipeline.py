"""
Локальный прогон демо: пары ideal+video из каталога данных без HTTP.
Каталог по умолчанию: DATA_DIR в этом файле (/app/data в контейнере).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from typing import Iterator, Literal, Optional

import cv2

from .ideal_storage import IdealImageStorage
from .processor import StorageViolationFrameProcessor
from .runner_common import prepare_ideal_bgr, run_video_pipeline

LOGGER = logging.getLogger("storage_violation_demo")

DATA_DIR = "/app/data"

DEMO_PRESETS: dict[int, tuple[str, str]] = {
    1: ("ideal.png", "video1.avi"),
    2: ("ofis_big.png", "ofis_big.mp4"),
    3: ("ofis_small.png", "ofis_small.mov"),
}

_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9._-]+$")
DEMO_CAMERA_ID = "demo_data"

IdealMode = Literal["static", "first_frame", "median", "mean"]


@contextmanager
def temporary_ideal_params(
    processor: StorageViolationFrameProcessor,
    *,
    ideal_mode: Optional[IdealMode] = None,
    ideal_frames: Optional[int] = None,
    bg_median_update_every: Optional[int] = None,
) -> Iterator[None]:
    """Временно меняет параметры фона на время демо-прогона."""
    if ideal_mode is None and ideal_frames is None and bg_median_update_every is None:
        yield
        return

    saved = {
        "ideal_mode": processor.ideal_mode,
        "ideal_frames": processor.ideal_frames,
        "bg_median_update_every": processor.bg_median_update_every,
    }
    processor.update_runtime_params(
        ideal_mode=ideal_mode,
        ideal_frames=ideal_frames,
        bg_median_update_every=bg_median_update_every,
    )
    try:
        yield
    finally:
        processor.update_runtime_params(**saved)


def sanitize_data_filename(name: str) -> str:
    """Только basename в каталоге данных, без путей (защита от path traversal)."""
    s = str(name).strip()
    if not s or len(s) > 255:
        raise ValueError("invalid filename")
    if ".." in s or "/" in s or "\\" in s:
        raise ValueError("only a basename is allowed, no folders or '..'")
    base = os.path.basename(s)
    if base != s:
        raise ValueError("only a basename is allowed")
    if not _SAFE_FILENAME.match(base):
        raise ValueError("filename: allowed characters are letters, digits, . _ -")
    return base


def run_demo_video_files(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    duration_sec: float,
    video_path: str,
    log_label: str,
    ideal_path: str | None = None,
) -> str:
    """
    Читает video_path, пишет MP4 во временный файл.
    При ideal_mode=static нужен ideal_path; иначе фон строится из кадров видео.
    duration_sec == -1 — весь ролик до EOF.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Missing video: {video_path}")

    ideal_bgr = None
    if processor.ideal_mode == "static":
        if ideal_path is None:
            raise ValueError("ideal_path is required when ideal_mode=static")
        if not os.path.isfile(ideal_path):
            raise FileNotFoundError(f"Missing ideal image: {ideal_path}")
        ideal_bgr = cv2.imread(ideal_path, cv2.IMREAD_COLOR)
        if ideal_bgr is None:
            raise ValueError(f"Failed to decode ideal image: {ideal_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 1000:
        fps = 25.0
    cap.release()

    if duration_sec < 0:
        if abs(duration_sec + 1.0) > 1e-6:
            raise ValueError("duration_sec must be positive or exactly -1 for the whole file")
        limit_duration: float | None = None
    else:
        limit_duration = float(duration_sec)

    camera_id = DEMO_CAMERA_ID
    processor.reset_camera(camera_id)
    ideal = prepare_ideal_bgr(processor, ideal_storage, camera_id, ideal_bgr)

    fd, out_path = tempfile.mkstemp(prefix="demo_", suffix=".mp4")
    os.close(fd)

    run_video_pipeline(
        processor,
        camera_id,
        ideal,
        video_path=video_path,
        duration_sec=limit_duration,
        output_video_path=out_path,
        synthetic_time_fps=fps,
    )

    LOGGER.info("[DEMO] %s -> %s (duration_sec=%s, fps=%.2f)", log_label, out_path, duration_sec, fps)
    return out_path


def run_demo_video(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    duration_sec: float,
    preset: int,
    data_dir: str | None = None,
) -> str:
    if preset not in DEMO_PRESETS:
        raise ValueError(f"preset must be 1..{len(DEMO_PRESETS)}, got {preset}")

    base = data_dir or DATA_DIR
    ideal_name, video_name = DEMO_PRESETS[preset]
    ideal_path = os.path.join(base, ideal_name) if processor.ideal_mode == "static" else None
    if processor.ideal_mode == "static":
        label = f"preset={preset} ({ideal_name}+{video_name})"
    else:
        label = f"preset={preset} ({video_name}, ideal_mode={processor.ideal_mode})"
    return run_demo_video_files(
        processor,
        ideal_storage,
        duration_sec,
        os.path.join(base, video_name),
        log_label=label,
        ideal_path=ideal_path,
    )


def run_demo_video_named(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    duration_sec: float,
    video_name: str,
    ideal_name: str | None = None,
    data_dir: str | None = None,
) -> str:
    in_video = sanitize_data_filename(video_name)
    base = data_dir or DATA_DIR
    ideal_path = None
    if processor.ideal_mode == "static":
        if ideal_name is None:
            raise ValueError("ideal_name is required when ideal_mode=static")
        in_ideal = sanitize_data_filename(ideal_name)
        ideal_path = os.path.join(base, in_ideal)
        label = f"custom ideal={in_ideal} video={in_video}"
    else:
        label = f"custom video={in_video} ideal_mode={processor.ideal_mode}"
    return run_demo_video_files(
        processor,
        ideal_storage,
        duration_sec,
        os.path.join(base, in_video),
        log_label=label,
        ideal_path=ideal_path,
    )
