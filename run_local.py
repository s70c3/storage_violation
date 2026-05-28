from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return np.ascontiguousarray(img)


def _draw_result(frame_bgr: np.ndarray, result: dict) -> np.ndarray:
    vis = frame_bgr.copy()

    status = bool(result.get("status", False))
    detected = bool(result.get("detected", False))
    debug = result.get("debug", {})

    candidate_boxes = result.get("candidate_boxes", []) or []
    reported_boxes = result.get("reported_boxes", []) or []

    for box in candidate_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            "candidate",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for box in reported_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            vis,
            "reported",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    text1 = f"detected={detected} status={status} cand={len(candidate_boxes)} reported={len(reported_boxes)}"
    text2 = (
        f"n_inst={debug.get('n_instances', 0)} "
        f"cand_dbg={debug.get('candidate_len', 0)} "
        f"rep_dbg={debug.get('reported_len', 0)} "
        f"alpha={debug.get('alpha_used', 0):.4f}"
    )
    cv2.putText(vis, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def _parse_polygons(polygons: Optional[str]) -> Optional[list[np.ndarray]]:
    if polygons is None or polygons.strip() == "":
        return None
    raw = json.loads(polygons)
    out: list[np.ndarray] = []
    for poly in raw:
        out.append(np.asarray(poly, dtype=np.float32))
    return out


def _build_processor(
    weights_path: str,
    device: str,
    half: bool,
    threshold: float,
    inp_ch: int,
    *,
    min_side: int,
    stationary_time_sec: float,
    max_long_side: int,
    visualization: bool,
    ideal_mode: str,
    bg_median_window: int,
    bg_median_update_every: int,
) -> "StorageViolationFrameProcessor":
    # Lazy imports so `--help` works without full runtime deps installed.
    from source.processor import StorageViolationFrameProcessor
    from source.segmentator import UNetVGG16Segmentator

    segmentator = UNetVGG16Segmentator(
        weights_path=weights_path,
        device=device,
        half=half,
        threshold=threshold,
        inp_ch=inp_ch,
    )
    processor = StorageViolationFrameProcessor(
        cd_segmentator=segmentator,
        min_side=min_side,
        stationary_time_sec=stationary_time_sec,
        max_long_side=max_long_side,
        visualization=visualization,
        ideal_mode=ideal_mode,
        bg_median_window=bg_median_window,
        bg_median_update_every=bg_median_update_every,
    )
    processor.load()
    return processor


def run_on_image(
    processor: "StorageViolationFrameProcessor",
    camera_id: str,
    ideal_bgr: Optional[np.ndarray],
    frame_path: str,
    polygons: Optional[list[np.ndarray]],
    out_json: Optional[str],
    out_image: Optional[str],
) -> dict:
    frame_bgr = _read_bgr(frame_path)
    result = processor.process_frame(
        camera_id=camera_id,
        frame_bgr=frame_bgr,
        ideal_bgr=ideal_bgr,
        polygons=polygons,
        now_mono=time.monotonic(),
    )

    if out_json:
        Path(out_json).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    if out_image:
        vis = _draw_result(frame_bgr, result)
        ok = cv2.imwrite(out_image, vis)
        if not ok:
            raise RuntimeError(f"Failed to write image: {out_image}")
    return result


def run_on_video(
    processor: "StorageViolationFrameProcessor",
    camera_id: str,
    ideal_bgr: Optional[np.ndarray],
    video_path: str,
    polygons: Optional[list[np.ndarray]],
    out_video: str,
    out_jsonl: Optional[str],
    every_n_frames: int,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_video}")

    jsonl_f = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
    try:
        frame_idx = 0
        last_result: dict = {
            "detected": False,
            "status": False,
            "candidate_boxes": [],
            "reported_boxes": [],
            "debug": {},
        }

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            if every_n_frames <= 1 or (frame_idx % every_n_frames) == 0:
                last_result = processor.process_frame(
                    camera_id=camera_id,
                    frame_bgr=frame_bgr,
                    ideal_bgr=ideal_bgr,
                    polygons=polygons,
                    now_mono=time.monotonic(),
                )
                if jsonl_f is not None:
                    jsonl_f.write(json.dumps({"frame_idx": frame_idx, "result": last_result}, ensure_ascii=False) + "\n")

            vis = _draw_result(frame_bgr, last_result)
            writer.write(vis)
    finally:
        cap.release()
        writer.release()
        if jsonl_f is not None:
            jsonl_f.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local runner for storage_violation pipeline (no API)")

    p.add_argument("--weights", type=str, default="weights/cd_weights.pt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--half", action="store_true", help="Use FP16 (only makes sense on CUDA)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--inp-ch", type=int, default=9)

    p.add_argument("--camera-id", type=str, default="cam_1")
    p.add_argument("--ideal-image", type=str, default=None)
    p.add_argument("--polygons", type=str, default=None, help='Polygons JSON, e.g. \'[[[100,100],[900,100],[900,700],[100,700]]]\'')

    p.add_argument("--min-side", type=int, default=10)
    p.add_argument("--stationary-time-sec", type=float, default=5.0)
    p.add_argument("--max-long-side", type=int, default=640)
    p.add_argument("--visualization", action="store_true")
    p.add_argument("--ideal-mode", type=str, default="static", choices=["static", "first_frame", "median"])
    p.add_argument("--bg-median-window", type=int, default=25)
    p.add_argument("--bg-median-update-every", type=int, default=5)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=str, default=None, help="Path to single frame image")
    src.add_argument("--video", type=str, default=None, help="Path to input video")

    p.add_argument("--out-json", type=str, default=None, help="Output JSON path (image mode)")
    p.add_argument("--out-image", type=str, default=None, help="Output image path (image mode)")

    p.add_argument("--out-video", type=str, default="output.mp4", help="Output video path (video mode)")
    p.add_argument("--out-jsonl", type=str, default=None, help="Output JSONL path (video mode)")
    p.add_argument("--every-n-frames", type=int, default=1, help="Run inference every Nth frame (video mode)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ideal_bgr = _read_bgr(args.ideal_image) if args.ideal_image else None
    polygons = _parse_polygons(args.polygons)

    try:
        processor = _build_processor(
            weights_path=args.weights,
            device=args.device,
            half=bool(args.half),
            threshold=float(args.threshold),
            inp_ch=int(args.inp_ch),
            min_side=int(args.min_side),
            stationary_time_sec=float(args.stationary_time_sec),
            max_long_side=int(args.max_long_side),
            visualization=bool(args.visualization),
            ideal_mode=str(args.ideal_mode),
            bg_median_window=int(args.bg_median_window),
            bg_median_update_every=int(args.bg_median_update_every),
        )
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise SystemExit(
            f"Не хватает зависимости для локального запуска: {missing}\n"
            f"Установите зависимости проекта (в частности, `supervision`) и повторите запуск."
        ) from e
    try:
        if args.frame:
            result = run_on_image(
                processor=processor,
                camera_id=args.camera_id,
                ideal_bgr=ideal_bgr,
                frame_path=args.frame,
                polygons=polygons,
                out_json=args.out_json,
                out_image=args.out_image,
            )
            print(json.dumps(result, ensure_ascii=False))
        else:
            run_on_video(
                processor=processor,
                camera_id=args.camera_id,
                ideal_bgr=ideal_bgr,
                video_path=args.video,
                polygons=polygons,
                out_video=args.out_video,
                out_jsonl=args.out_jsonl,
                every_n_frames=max(1, int(args.every_n_frames)),
            )
            print(f"[OK] Saved video: {args.out_video}")
            if args.out_jsonl:
                print(f"[OK] Saved jsonl: {args.out_jsonl}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()

