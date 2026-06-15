import argparse
import cv2
import json
import os
import requests

from source.runner_common import draw_result


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