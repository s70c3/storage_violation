import cv2
import json
import requests


API_URL = "http://127.0.0.1:8000"
CAMERA_ID = "cam_1"

IDEAL_IMAGE_PATH = "data/ideal.png"
VIDEO_PATH = "data/video1.avi"

POLYGONS = None
SEND_EVERY_N_FRAMES = 1
REQUEST_TIMEOUT = 60


def upload_ideal(api_url: str, camera_id: str, ideal_path: str) -> None:
    with open(ideal_path, "rb") as f:
        resp = requests.post(
            f"{api_url}/ideal/{camera_id}",
            files={"image": (ideal_path, f, "image/jpeg")},
            timeout=REQUEST_TIMEOUT,
        )
    print("UPLOAD IDEAL:", resp.status_code, resp.text)
    resp.raise_for_status()


def send_frame(api_url: str, camera_id: str, frame_bgr, polygons=None):
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
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def draw_result(frame_bgr, result: dict):
    vis = frame_bgr.copy()

    status = bool(result.get("status", False))
    detected = bool(result.get("detected", False))
    debug = result.get("debug", {})

    print(result)
    candidate_boxes = result.get("candidate_boxes", [])
    reported_boxes = result.get("reported_boxes", [])

    # Зеленые: видны сейчас, но еще не стоят достаточно долго
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

    # Красные: стоят уже достаточно долго
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

    cv2.putText(vis, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    return vis


def main():
    upload_ideal(API_URL, CAMERA_ID, IDEAL_IMAGE_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Writing result video to: {output_path}")

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

        if frame_idx % SEND_EVERY_N_FRAMES == 0:
            try:
                last_result = send_frame(
                    api_url=API_URL,
                    camera_id=CAMERA_ID,
                    frame_bgr=frame,
                    polygons=POLYGONS,
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