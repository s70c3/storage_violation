from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import zipfile
from typing import Any, Optional

import cv2
import numpy as np

from .ideal_storage import IdealImageStorage
from .processor import StorageViolationFrameProcessor
from .schemas import ModelParams, RuntimeParamsRequest, RuntimeParamsResponse
from .segmentator import UNetVGG16Segmentator


def draw_result(
    frame_bgr: np.ndarray,
    result: dict,
    frame_idx: int | None = None,
) -> np.ndarray:
    vis = frame_bgr.copy()
    h, _w = vis.shape[:2]

    status = bool(result.get("status", False))
    detected = bool(result.get("detected", False))
    debug = result.get("debug", {})

    candidate_boxes = result.get("candidate_boxes", []) or []
    reported_boxes = result.get("reported_boxes", []) or []
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
    cv2.putText(vis, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    if frame_idx is not None:
        cv2.putText(
            vis,
            f"frame {frame_idx}",
            (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


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
    if isinstance(value, list):
        out: list[list[int]] = []
        for box in value:
            b = np.asarray(box, dtype=np.float64).reshape(-1)
            if b.size >= 4:
                out.append([int(b[i]) for i in range(4)])
        return out
    return []


def process_frame_to_api_dict(result: dict) -> dict[str, Any]:
    candidate_boxes = boxes_to_list(
        result.get("candidate_boxes", result.get("pending_candidate_boxes"))
    )
    reported_boxes = boxes_to_list(result.get("reported_boxes", result.get("boxes")))
    cand_ids = [int(x) for x in (result.get("candidate_track_ids") or [])]
    rep_ids = [int(x) for x in (result.get("reported_track_ids") or [])]
    return {
        "detected": bool(result.get("detected", False)),
        "status": bool(result.get("status", False)),
        "candidate_boxes": candidate_boxes,
        "reported_boxes": reported_boxes,
        "candidate_track_ids": cand_ids,
        "reported_track_ids": rep_ids,
        "debug": result.get("debug", {}),
    }


def parse_polygons_json(polygons_json: Optional[str]) -> Optional[list[np.ndarray]]:
    if polygons_json is None or str(polygons_json).strip() == "":
        return None

    try:
        polygons_raw = json.loads(polygons_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid polygons JSON: {e}") from e

    if polygons_raw is None:
        return None
    if not isinstance(polygons_raw, list):
        raise ValueError("polygons must be a JSON list")

    out: list[np.ndarray] = []
    for poly in polygons_raw:
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Each polygon must have shape [N,2]")
        out.append(arr)
    return out


def resolve_ideal_bgr(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    camera_id: str,
    ideal_bgr_override: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if ideal_bgr_override is not None and processor.ideal_mode == "static":
        return ideal_bgr_override
    if processor.ideal_mode != "static":
        return None
    ideal_bgr = ideal_storage.load(camera_id)
    if ideal_bgr is None:
        raise ValueError(
            f"Ideal image for camera_id={camera_id} not found. "
            f"Upload via POST /ideal/{camera_id} or set ideal_mode to first_frame/median/mean."
        )
    return ideal_bgr


def prepare_ideal_bgr(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    camera_id: str,
    ideal_bgr_override: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if ideal_bgr_override is not None:
        ideal_storage.save(camera_id, ideal_bgr_override)
    return resolve_ideal_bgr(processor, ideal_storage, camera_id, ideal_bgr_override)


def build_processor(
    *,
    weights_path: str,
    device: str,
    half: bool,
    threshold: float,
    inp_ch: int,
    min_side: int = 10,
    stationary_time_sec: float = 5.0,
    max_long_side: int = 640,
    visualization: bool = False,
    ideal_mode: str = "static",
    ideal_frames: int = 25,
    bg_median_update_every: int = 5,
    logger: logging.Logger | None = None,
) -> StorageViolationFrameProcessor:
    segmentator = UNetVGG16Segmentator(
        weights_path=weights_path,
        device=device,
        half=half,
        threshold=threshold,
        inp_ch=inp_ch,
        logger=logger,
    )
    processor = StorageViolationFrameProcessor(
        cd_segmentator=segmentator,
        min_side=min_side,
        stationary_time_sec=stationary_time_sec,
        max_long_side=max_long_side,
        visualization=visualization,
        ideal_mode=ideal_mode,
        ideal_frames=ideal_frames,
        bg_median_update_every=bg_median_update_every,
        logger=logger,
    )
    processor.load()
    return processor


def run_on_image(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    camera_id: str,
    frame_bgr: np.ndarray,
    ideal_bgr_override: Optional[np.ndarray] = None,
    polygons: Optional[list[np.ndarray]] = None,
) -> dict[str, Any]:
    ideal_bgr = prepare_ideal_bgr(processor, ideal_storage, camera_id, ideal_bgr_override)
    result = processor.process_frame(
        camera_id=camera_id,
        frame_bgr=frame_bgr,
        ideal_bgr=ideal_bgr,
        polygons=polygons,
        now_mono=time.monotonic(),
    )
    return process_frame_to_api_dict(result)


def build_runtime_params_response(
    processor: StorageViolationFrameProcessor,
    model_cfg: dict[str, Any],
) -> RuntimeParamsResponse:
    seg = processor.cd_segmentator
    return RuntimeParamsResponse(
        min_side=processor.min_side,
        stationary_time_sec=processor.stationary_time_sec,
        max_long_side=processor.max_long_side,
        ideal_mode=processor.ideal_mode,  # type: ignore[arg-type]
        ideal_frames=processor.ideal_frames,
        bg_median_window=processor.ideal_frames,
        bg_median_update_every=processor.bg_median_update_every,
        thr=processor.thr,
        morph_ksize=processor.morph_ksize,
        ema_tau_sec=processor.ema_tau_sec,
        ema_min_alpha=processor.ema_min_alpha,
        ema_max_alpha=processor.ema_max_alpha,
        recent_update_every=processor.recent_update_every,
        visualization=processor.visualization,
        tracker_track_activation_threshold=processor.tracker_track_activation_threshold,
        tracker_lost_track_buffer=processor.tracker_lost_track_buffer,
        tracker_minimum_matching_threshold=processor.tracker_minimum_matching_threshold,
        tracker_frame_rate=processor.tracker_frame_rate,
        tracker_minimum_consecutive_frames=processor.tracker_minimum_consecutive_frames,
        tracker_max_center_shift_px=processor.tracker_max_center_shift_px,
        tracker_max_center_shift_norm=processor.tracker_max_center_shift_norm,
        model=ModelParams(
            weights_path=str(model_cfg["weights_path"]),
            device=str(getattr(seg, "_device", model_cfg["device"])),
            threshold=float(getattr(seg, "threshold", model_cfg["threshold"])),
            half=bool(getattr(seg, "half", model_cfg["half"])),
            inp_ch=int(getattr(seg, "inp_ch", model_cfg["inp_ch"])),
        ),
    )


def apply_model_config_updates(
    processor: StorageViolationFrameProcessor,
    model_cfg: dict[str, Any],
    body: RuntimeParamsRequest,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or logging.getLogger("storage_violation")
    new_cfg = dict(model_cfg)
    for key in ("weights_path", "device", "threshold", "half", "inp_ch"):
        val = getattr(body, key, None)
        if val is not None:
            new_cfg[key] = val

    reload_needed = any(
        new_cfg[k] != model_cfg[k] for k in ("weights_path", "device", "half", "inp_ch")
    )
    threshold_changed = new_cfg["threshold"] != model_cfg["threshold"]

    if reload_needed:
        processor.cd_segmentator.close()
        segmentator = UNetVGG16Segmentator(
            weights_path=str(new_cfg["weights_path"]),
            device=str(new_cfg["device"]),
            half=bool(new_cfg["half"]),
            threshold=float(new_cfg["threshold"]),
            inp_ch=int(new_cfg["inp_ch"]),
            logger=log,
        )
        segmentator.load()
        processor.cd_segmentator = segmentator
        log.info(f"[MODEL] reloaded device={segmentator._device} half={segmentator.half}")
    elif threshold_changed:
        processor.cd_segmentator.threshold = float(new_cfg["threshold"])
        log.info(f"[MODEL] threshold={new_cfg['threshold']}")

    return new_cfg


def apply_processor_params_from_request(
    processor: StorageViolationFrameProcessor,
    body: RuntimeParamsRequest,
) -> None:
    stationary_time_sec = body.stationary_time_sec
    if stationary_time_sec is None and body.threshold_time_sec is not None:
        stationary_time_sec = body.threshold_time_sec

    processor.update_runtime_params(
        min_side=body.min_side,
        stationary_time_sec=stationary_time_sec,
        max_long_side=body.max_long_side,
        ideal_mode=body.ideal_mode,
        ideal_frames=body.ideal_frames,
        bg_median_window=body.bg_median_window,
        bg_median_update_every=body.bg_median_update_every,
        thr=body.thr,
        morph_ksize=body.morph_ksize,
        ema_tau_sec=body.ema_tau_sec,
        ema_min_alpha=body.ema_min_alpha,
        ema_max_alpha=body.ema_max_alpha,
        recent_update_every=body.recent_update_every,
        visualization=body.visualization,
        tracker_track_activation_threshold=body.tracker_track_activation_threshold,
        tracker_lost_track_buffer=body.tracker_lost_track_buffer,
        tracker_minimum_matching_threshold=body.tracker_minimum_matching_threshold,
        tracker_frame_rate=body.tracker_frame_rate,
        tracker_minimum_consecutive_frames=body.tracker_minimum_consecutive_frames,
        tracker_max_center_shift_px=body.tracker_max_center_shift_px,
        tracker_max_center_shift_norm=body.tracker_max_center_shift_norm,
    )


def open_video_capture(src: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(src)


def _empty_last_result() -> dict:
    return {
        "detected": False,
        "status": False,
        "candidate_boxes": [],
        "reported_boxes": [],
        "candidate_track_ids": [],
        "reported_track_ids": [],
        "debug": {},
    }


def _frame_limit_from_duration(
    duration_sec: Optional[float],
    fps: float,
    max_frames: Optional[int],
) -> Optional[int]:
    frame_limit = None
    if max_frames is not None and int(max_frames) > 0:
        frame_limit = int(max_frames)
    if duration_sec is not None and float(duration_sec) > 0:
        by_duration = max(1, int(round(float(duration_sec) * float(fps))))
        frame_limit = min(frame_limit, by_duration) if frame_limit else by_duration
    return frame_limit


def run_video_pipeline(
    processor: StorageViolationFrameProcessor,
    camera_id: str,
    ideal_bgr: Optional[np.ndarray],
    *,
    video_path: Optional[str] = None,
    rtsp_url: Optional[str] = None,
    polygons: Optional[list[np.ndarray]] = None,
    every_n_frames: int = 1,
    reconnect_sec: float = 1.0,
    max_frames: Optional[int] = None,
    duration_sec: Optional[float] = None,
    include_jsonl: bool = False,
    output_video_path: Optional[str] = None,
    output_jsonl_path: Optional[str] = None,
    show: bool = False,
    window_name: str = "storage_violation",
    wait_ms: int = 1,
    synthetic_time_fps: Optional[float] = None,
) -> tuple[str, Optional[str]]:
    """Process video file or RTSP; write annotated MP4. Returns (video_path, jsonl_path_or_none)."""
    src = rtsp_url if rtsp_url else video_path
    if not src:
        raise ValueError("video source is empty")

    cap = open_video_capture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 1000:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions")

    time_fps = float(synthetic_time_fps) if synthetic_time_fps else fps
    frame_limit = _frame_limit_from_duration(duration_sec, time_fps, max_frames)

    if output_video_path is not None:
        out_video_path = output_video_path
    else:
        out_video_fd, out_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_video_fd)

    jsonl_path: Optional[str] = None
    if output_jsonl_path is not None:
        jsonl_path = output_jsonl_path
    elif include_jsonl:
        jsonl_fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(jsonl_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        safe_unlink(out_video_path)
        if jsonl_path:
            safe_unlink(jsonl_path)
        raise RuntimeError("Failed to open video writer")

    jsonl_f = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None
    every_n = max(1, int(every_n_frames))
    frame_idx = 0

    try:
        last_result = _empty_last_result()

        while True:
            if frame_limit is not None and frame_idx >= frame_limit:
                break

            ok, frame_bgr = cap.read()
            if not ok:
                if rtsp_url:
                    cap.release()
                    time.sleep(max(0.0, float(reconnect_sec)))
                    cap = open_video_capture(src)
                    if not cap.isOpened():
                        continue
                    continue
                break

            frame_idx += 1
            if (frame_idx % every_n) == 0:
                now_mono = (
                    frame_idx / time_fps
                    if synthetic_time_fps is not None
                    else time.monotonic()
                )
                raw = processor.process_frame(
                    camera_id=camera_id,
                    frame_bgr=frame_bgr,
                    ideal_bgr=ideal_bgr,
                    polygons=polygons,
                    now_mono=now_mono,
                )
                last_result = process_frame_to_api_dict(raw)
                if jsonl_f is not None:
                    jsonl_f.write(
                        json.dumps({"frame_idx": frame_idx, "result": last_result}, ensure_ascii=False)
                        + "\n"
                    )

            vis = draw_result(frame_bgr, last_result, frame_idx=frame_idx)
            writer.write(vis)

            if show:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(int(wait_ms)) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        writer.release()
        if show:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                cv2.destroyAllWindows()
        if jsonl_f is not None:
            jsonl_f.close()

    if frame_idx == 0:
        safe_unlink(out_video_path)
        if jsonl_path:
            safe_unlink(jsonl_path)
        raise RuntimeError("No frames read from video (empty or unreadable)")

    return out_video_path, jsonl_path


def pack_output_zip(video_path: str, jsonl_path: str, output_name: str) -> str:
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(zip_fd)
    video_name = output_name if output_name.endswith(".mp4") else f"{output_name}.mp4"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(video_path, arcname=video_name)
        zf.write(jsonl_path, arcname="results.jsonl")
    return zip_path


def encode_image_bgr_jpeg_b64(frame_bgr: np.ndarray) -> str:
    import base64

    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode annotated image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def safe_unlink_many(*paths: Optional[str]) -> None:
    for path in paths:
        if path:
            safe_unlink(path)
