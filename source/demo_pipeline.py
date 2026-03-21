"""
Локальный прогон демо: пары ideal+video из каталога данных без HTTP.
Каталог по умолчанию: DATA_DIR в этом файле (/app/data в контейнере).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any

import cv2
import numpy as np

from .ideal_storage import IdealImageStorage
from .processor import StorageViolationFrameProcessor

LOGGER = logging.getLogger("storage_violation_demo")

# Каталог с ideal/видео для демо (в Docker: volume ./data -> /app/data)
DATA_DIR = "/app/data"

# Пресеты: положите файлы в DATA_DIR (см. README)
DEMO_PRESETS: dict[int, tuple[str, str]] = {
    1: ("ideal.png", "video1.avi"),
    2: ("ofis_big.png", "ofis_big.mp4"),
    3: ("ofis_small.png", "ofis_small.mov"),
}

_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9._-]+$")


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

DEMO_CAMERA_ID = "demo_data"


def _empty_draw_result() -> dict[str, Any]:
    return {
        "detected": False,
        "status": False,
        "candidate_boxes": [],
        "reported_boxes": [],
        "candidate_track_ids": [],
        "reported_track_ids": [],
        "debug": {},
    }


def result_to_draw_dict(result: dict) -> dict[str, Any]:
    """Приводит ответ process_frame к виду, ожидаемому draw_result."""

    def boxes_to_list(value) -> list[list[int]]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return []
            arr = np.asarray(value, dtype=np.int32)
            if arr.ndim == 1:
                return [arr.tolist()] if arr.size >= 4 else []
            return arr.tolist()
        out: list[list[int]] = []
        for box in value:
            b = np.asarray(box, dtype=np.float64).reshape(-1)
            if b.size >= 4:
                out.append([int(b[i]) for i in range(4)])
        return out

    def ids_to_list(value) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
        return []

    cand = boxes_to_list(result.get("candidate_boxes", result.get("pending_candidate_boxes")))
    rep = boxes_to_list(result.get("reported_boxes", result.get("boxes")))
    cand_ids = ids_to_list(result.get("candidate_track_ids"))
    rep_ids = ids_to_list(result.get("reported_track_ids"))
    # выравнивание длин по числу боксов
    while len(cand_ids) < len(cand):
        cand_ids.append(-1)
    while len(rep_ids) < len(rep):
        rep_ids.append(-1)
    cand_ids = cand_ids[: len(cand)]
    rep_ids = rep_ids[: len(rep)]

    return {
        "detected": bool(result.get("detected", False)),
        "status": bool(result.get("status", False)),
        "candidate_boxes": cand,
        "reported_boxes": rep,
        "candidate_track_ids": cand_ids,
        "reported_track_ids": rep_ids,
        "debug": result.get("debug", {}),
    }


def draw_result(
    frame_bgr: np.ndarray,
    result: dict,
    frame_idx: int,
) -> np.ndarray:
    """Отрисовка: зелёный candidate, красный reported, id трека, номер кадра внизу слева."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    status = bool(result.get("status", False))
    detected = bool(result.get("detected", False))
    debug = result.get("debug", {})

    candidate_boxes = result.get("candidate_boxes", [])
    reported_boxes = result.get("reported_boxes", [])
    cand_ids = result.get("candidate_track_ids") or []
    rep_ids = result.get("reported_track_ids") or []

    for i, box in enumerate(candidate_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tid = cand_ids[i] if i < len(cand_ids) else -1
        label = f"cand id={tid}" if tid >= 0 else "candidate"
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for i, box in enumerate(reported_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        tid = rep_ids[i] if i < len(rep_ids) else -1
        label = f"rep id={tid}" if tid >= 0 else "reported"
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    text1 = (
        f"detected={detected} status={status} "
        f"cand={len(candidate_boxes)} reported={len(reported_boxes)}"
    )
    alpha = debug.get("alpha_used", 0)
    try:
        alpha_f = float(alpha) if alpha is not None else 0.0
    except (TypeError, ValueError):
        alpha_f = 0.0
    text2 = (
        f"n_inst={debug.get('n_instances', 0)} "
        f"cand_dbg={debug.get('candidate_len', 0)} "
        f"rep_dbg={debug.get('reported_len', 0)} "
        f"alpha={alpha_f:.4f}"
    )

    cv2.putText(
        vis, text1, (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        vis, text2, (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
    )

    frame_label = f"frame {frame_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.7, 2
    (tw, th_text), bl = cv2.getTextSize(frame_label, font, fs, th)
    y_bl = h - 8
    cv2.putText(
        vis,
        frame_label,
        (10, y_bl),
        font,
        fs,
        (200, 200, 255),
        th,
        cv2.LINE_AA,
    )

    return vis


def run_demo_video_files(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    duration_sec: float,
    ideal_path: str,
    video_path: str,
    log_label: str,
) -> str:
    """
    Читает ideal_path / video_path, пишет MP4 во временный файл.

    duration_sec == -1 — весь ролик до EOF.
    Время кадра: now_mono = frame_index / fps.
    """
    if not os.path.isfile(ideal_path):
        raise FileNotFoundError(f"Missing ideal image: {ideal_path}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Missing video: {video_path}")

    ideal_bgr = cv2.imread(ideal_path, cv2.IMREAD_COLOR)
    if ideal_bgr is None:
        raise ValueError(f"Failed to decode ideal image: {ideal_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 1000:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions")

    if duration_sec < 0:
        if abs(duration_sec + 1.0) > 1e-6:
            raise ValueError("duration_sec must be positive or exactly -1 for the whole file")
        unlimited = True
    else:
        unlimited = False

    if unlimited:
        max_frames: int | None = None
    else:
        max_frames = max(1, int(round(float(duration_sec) * fps)))

    camera_id = DEMO_CAMERA_ID
    processor.reset_camera(camera_id)
    ideal_storage.save(camera_id, ideal_bgr)

    fd, out_path = tempfile.mkstemp(prefix="demo_", suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        os.unlink(out_path)
        raise RuntimeError("Failed to create output video writer")

    frame_idx = 0

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            now_mono = frame_idx / fps

            result = processor.process_frame(
                camera_id=camera_id,
                frame_bgr=frame,
                ideal_bgr=ideal_bgr,
                polygons=None,
                now_mono=now_mono,
            )
            last_draw = result_to_draw_dict(result)
            vis = draw_result(frame, last_draw, frame_idx)
            writer.write(vis)
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise RuntimeError("No frames read from video (empty or unreadable)")

    LOGGER.info(
        "[DEMO] %s wrote %d frames -> %s (duration_sec=%s, fps=%.2f)",
        log_label,
        frame_idx,
        out_path,
        duration_sec,
        fps,
    )
    return out_path


def run_demo_video(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    duration_sec: float,
    preset: int,
    data_dir: str | None = None,
) -> str:
    """Пресет 1..3 — см. DEMO_PRESETS."""
    if preset not in DEMO_PRESETS:
        raise ValueError(f"preset must be 1..{len(DEMO_PRESETS)}, got {preset}")

    base = data_dir or DATA_DIR
    ideal_name, video_name = DEMO_PRESETS[preset]
    ideal_path = os.path.join(base, ideal_name)
    video_path = os.path.join(base, video_name)
    return run_demo_video_files(
        processor,
        ideal_storage,
        duration_sec,
        ideal_path,
        video_path,
        log_label=f"preset={preset} ({ideal_name}+{video_name})",
    )


def run_demo_video_named(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    duration_sec: float,
    ideal_name: str,
    video_name: str,
    data_dir: str | None = None,
) -> str:
    """Произвольные имена файлов в каталоге данных (только basename, без подпапок)."""
    in_ideal = sanitize_data_filename(ideal_name)
    in_video = sanitize_data_filename(video_name)
    base = data_dir or DATA_DIR
    ideal_path = os.path.join(base, in_ideal)
    video_path = os.path.join(base, in_video)
    return run_demo_video_files(
        processor,
        ideal_storage,
        duration_sec,
        ideal_path,
        video_path,
        log_label=f"custom ideal={in_ideal} video={in_video}",
    )
