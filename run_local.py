from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from source.ideal_storage import IdealImageStorage
from source.runner_common import (
    build_processor,
    draw_result,
    parse_polygons_json,
    prepare_ideal_bgr,
    run_on_image,
    run_video_pipeline,
)


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return np.ascontiguousarray(img)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local runner for storage_violation pipeline (no API)")

    p.add_argument("--weights", type=str, default="weights/cd_weights.pt")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, cuda:0, mps",
    )
    p.add_argument("--half", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--inp-ch", type=int, default=9)

    p.add_argument("--camera-id", type=str, default="cam_1")
    p.add_argument("--ideal-image", type=str, default=None)
    p.add_argument("--polygons", type=str, default=None)

    p.add_argument("--min-side", type=int, default=10)
    p.add_argument("--stationary-time-sec", type=float, default=5.0)
    p.add_argument("--max-long-side", type=int, default=640)
    p.add_argument("--visualization", action="store_true")
    p.add_argument(
        "--ideal-mode",
        type=str,
        default="static",
        choices=["static", "first_frame", "median", "mean"],
    )
    p.add_argument("--ideal-frames", type=int, default=25)
    p.add_argument("--bg-median-window", type=int, default=None)
    p.add_argument("--bg-median-update-every", type=int, default=5)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=str, default=None)
    src.add_argument("--video", type=str, default=None)
    src.add_argument("--rtsp", type=str, default=None)

    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-image", type=str, default=None)
    p.add_argument("--out-video", type=str, default="output.mp4")
    p.add_argument("--output-name", type=str, default=None)
    p.add_argument("--out-jsonl", type=str, default=None)
    p.add_argument("--every-n-frames", type=int, default=1)
    p.add_argument("--show", action="store_true")
    p.add_argument("--window-name", type=str, default="storage_violation")
    p.add_argument("--wait-ms", type=int, default=1)
    p.add_argument("--rtsp-reconnect-sec", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--duration-sec", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ideal_bgr = _read_bgr(args.ideal_image) if args.ideal_image else None
    polygons = parse_polygons_json(args.polygons)
    ideal_storage = IdealImageStorage()
    if ideal_bgr is not None:
        ideal_storage.save(args.camera_id, ideal_bgr)

    try:
        processor = build_processor(
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
            ideal_frames=int(
                args.bg_median_window if args.bg_median_window is not None else args.ideal_frames
            ),
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
            frame_bgr = _read_bgr(args.frame)
            payload = run_on_image(
                processor=processor,
                ideal_storage=ideal_storage,
                camera_id=args.camera_id,
                frame_bgr=frame_bgr,
                ideal_bgr_override=ideal_bgr,
                polygons=polygons,
            )
            if args.out_json:
                Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            if args.out_image:
                vis = draw_result(frame_bgr, payload)
                if not cv2.imwrite(args.out_image, vis):
                    raise RuntimeError(f"Failed to write image: {args.out_image}")
            print(json.dumps(payload, ensure_ascii=False))
        else:
            out_video = args.output_name or args.out_video
            ideal = prepare_ideal_bgr(processor, ideal_storage, args.camera_id, ideal_bgr)
            run_video_pipeline(
                processor,
                args.camera_id,
                ideal,
                video_path=args.video,
                rtsp_url=args.rtsp,
                polygons=polygons,
                every_n_frames=max(1, int(args.every_n_frames)),
                reconnect_sec=float(args.rtsp_reconnect_sec),
                max_frames=args.max_frames,
                duration_sec=args.duration_sec,
                include_jsonl=args.out_jsonl is not None,
                output_video_path=out_video,
                output_jsonl_path=args.out_jsonl,
                show=bool(args.show),
                window_name=str(args.window_name),
                wait_ms=int(args.wait_ms),
            )
            print(f"[OK] Saved video: {out_video}")
            if args.out_jsonl:
                print(f"[OK] Saved jsonl: {args.out_jsonl}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()
