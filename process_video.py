import argparse
import cv2
import json
import os
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_CAMERA_ID = "cam_1"
DEFAULT_IDEAL_IMAGE_PATH = "data/ideal.png"
DEFAULT_VIDEO_PATH = "data/video1.avi"
DEFAULT_OUTPUT_PATH = "output.mp4"
DEFAULT_SEND_EVERY_N_FRAMES = 1
DEFAULT_REQUEST_TIMEOUT = 60


def parse_args():
    parser = argparse.ArgumentParser(description="Send video frames to storage violation API")

    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID, help="Camera ID")
    parser.add_argument("--ideal-image", type=str, default=DEFAULT_IDEAL_IMAGE_PATH, help="Path to ideal image")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to output video")
    parser.add_argument(
        "--send-every-n-frames",
        type=int,
        default=DEFAULT_SEND_EVERY_N_FRAMES,
        help="Send every N-th frame to API",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--polygons",
        type=str,
        default=None,
        help='Polygons as JSON string, e.g. \'[[[100,100],[900,100],[900,700],[100,700]]]\'',
    )

    return parser.parse_args()


def upload_ideal(api_url: str, camera_id: str, ideal_path: str, request_timeout: int) -> None:
    with open(ideal_path, "rb") as f:
        resp = requests.post(
            f"{api_url}/ideal/{camera_id}",
            files={"image": (os.path.basename(ideal_path), f, "image/jpeg")},
            timeout=request_timeout,
        )
    print("UPLOAD IDEAL:", resp.status_code, resp.text)
    resp.raise_for_status()


def send_frame(
    api_url: str,
    camera_id: str,
    frame_bgr,
    request_timeout: int,
    polygons=None,
):
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")

    data = {"camera_id": camera_id}
    if polygons is not None:
        data["polygons"] = json.dumps(polygons)

    files = {
        "frame": ("frame.jpg", encoded.tobytes(), "image/jpeg"),
    }

    resp = requests.post(
        f"{api_url}/process_frame",
        data=data,
        files=files,
        timeout=request_timeout,
    )
    resp.raise_for_status()
    return resp.json()


def draw_result(frame_bgr, result: dict):
    vis = frame_bgr.copy()

    status = bool(result.get("status", False))
    detected = bool(result.get("detected", False))
    debug = result.get("debug", {})

    candidate_boxes = result.get("candidate_boxes", [])
    reported_boxes = result.get("reported_boxes", [])

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

    text1 = (
        f"detected={detected} status={status} "
        f"cand={len(candidate_boxes)} reported={len(reported_boxes)}"
    )
    text2 = (
        f"n_inst={debug.get('n_instances', 0)} "
        f"cand_dbg={debug.get('candidate_len', 0)} "
        f"rep_dbg={debug.get('reported_len', 0)} "
        f"alpha={debug.get('alpha_used', 0):.4f}"
    )

    cv2.putText(
        vis, text1, (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        vis, text2, (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
    )

    return vis


def main():
    args = parse_args()

    polygons = None
    if args.polygons:
        polygons = json.loads(args.polygons)

    upload_ideal(
        api_url=args.api_url,
        camera_id=args.camera_id,
        ideal_path=args.ideal_image,
        request_timeout=args.request_timeout,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"[INFO] Input video: {args.video}")
    print(f"[INFO] Output video: {args.output}")
    print(f"[INFO] Writing result video to: {args.output}")

    frame_idx = 0
    last_result = {
        "detected": False,
        "status": False,
        "boxes": [],
        "candidate_boxes": [],
        "reported_boxes": [],
        "debug": {},
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        if frame_idx % args.send_every_n_frames == 0:
            try:
                last_result = send_frame(
                    api_url=args.api_url,
                    camera_id=args.camera_id,
                    frame_bgr=frame,
                    polygons=polygons,
                    request_timeout=args.request_timeout,
                )
                print(f"frame={frame_idx} -> {last_result}")
            except requests.RequestException as e:
                print(f"Request failed on frame {frame_idx}: {e}")

        vis = draw_result(frame, last_result)
        writer.write(vis)

    cap.release()
    writer.release()

    print("[INFO] Done. Video saved.")


if __name__ == "__main__":
    main()