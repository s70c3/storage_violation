from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .demo_pipeline import run_demo_video, run_demo_video_named
from .ideal_storage import IdealImageStorage
from .processor import StorageViolationFrameProcessor
from .segmentator import UNetVGG16Segmentator

LOGGER = logging.getLogger("storage_violation_api")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)


MODEL_WEIGHTS_PATH = "weights/cd_weights.pt"
MODEL_DEVICE = "auto"   # auto | cpu | cuda:0 | mps
MODEL_THRESHOLD = 0.5
MODEL_HALF = True
MODEL_INP_CH = 9

PROCESSOR_KWARGS = dict(
    min_side=10,
    stationary_time_sec=5.0,
    max_long_side=640,
    ideal_mode="static",
)

def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode uploaded image")
    return img


async def read_uploaded_image(file: UploadFile) -> np.ndarray:
    if file is None:
        raise HTTPException(status_code=400, detail="Image file is required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return decode_image_bytes(content)


def normalize_polygons_from_json(polygons_json: Optional[str]) -> Optional[list[np.ndarray]]:
    if polygons_json is None or polygons_json.strip() == "":
        return None

    try:
        polygons_raw = json.loads(polygons_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid polygons JSON: {e}")

    if polygons_raw is None:
        return None

    if not isinstance(polygons_raw, list):
        raise HTTPException(status_code=400, detail="polygons must be a JSON list")

    out: list[np.ndarray] = []
    for poly in polygons_raw:
        arr = np.asarray(poly, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise HTTPException(
                status_code=400,
                detail="Each polygon must have shape [N,2]",
            )
        out.append(arr)

    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("[API] Loading model and processor...")

    segmentator = UNetVGG16Segmentator(
        weights_path=MODEL_WEIGHTS_PATH,
        device=MODEL_DEVICE,
        half=MODEL_HALF,
        threshold=MODEL_THRESHOLD,
        inp_ch=MODEL_INP_CH,
        logger=LOGGER,
    )
    processor = StorageViolationFrameProcessor(
        cd_segmentator=segmentator,
        logger=LOGGER,
        **PROCESSOR_KWARGS,
    )
    processor.load()

    ideal_storage = IdealImageStorage()

    app.state.processor = processor
    app.state.ideal_storage = ideal_storage

    LOGGER.info("[API] Processor is ready")
    yield

    LOGGER.info("[API] Shutting down processor...")
    processor.close()
    ideal_storage.clear()
    LOGGER.info("[API] Shutdown complete")


app = FastAPI(
    title="Storage Violation API",
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ideal/{camera_id}")
async def upload_ideal(
    camera_id: str,
    image: UploadFile = File(...),
):
    ideal_storage: IdealImageStorage = app.state.ideal_storage

    image_bgr = await read_uploaded_image(image)
    ideal_storage.save(camera_id, image_bgr)

    return {
        "camera_id": camera_id,
        "stored": True,
        "filename": image.filename,
        "shape": list(image_bgr.shape),
    }


@app.get("/ideal/{camera_id}")
def get_ideal_info(camera_id: str):
    ideal_storage: IdealImageStorage = app.state.ideal_storage
    exists = ideal_storage.exists(camera_id)
    return {
        "camera_id": camera_id,
        "exists": exists,
    }


@app.get("/ideal")
def list_ideal_cameras():
    ideal_storage: IdealImageStorage = app.state.ideal_storage
    camera_ids = ideal_storage.list_camera_ids()
    return {
        "camera_ids": camera_ids,
        "count": len(camera_ids),
    }


@app.delete("/ideal/{camera_id}")
def delete_ideal(camera_id: str):
    ideal_storage: IdealImageStorage = app.state.ideal_storage
    deleted = ideal_storage.delete(camera_id)
    return {
        "camera_id": camera_id,
        "deleted": deleted,
    }

@app.post("/process_frame")
async def process_frame(
    camera_id: str = Form(...),
    frame: UploadFile = File(...),
    polygons: Optional[str] = Form(None),
):
    processor: StorageViolationFrameProcessor = app.state.processor
    ideal_storage: IdealImageStorage = app.state.ideal_storage

    ideal_bgr = ideal_storage.load(camera_id)
    if ideal_bgr is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ideal image for camera_id={camera_id} not found. "
                f"Upload it first via POST /ideal/{camera_id}"
            ),
        )

    frame_bgr = await read_uploaded_image(frame)
    polygons_np = normalize_polygons_from_json(polygons)

    result = processor.process_frame(
        camera_id=camera_id,
        frame_bgr=frame_bgr,
        ideal_bgr=ideal_bgr,
        polygons=polygons_np,
    )

    def boxes_to_list(value) -> list[list[int]]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.astype(int).tolist() if value.size > 0 else []
        if isinstance(value, list):
            return [[int(v) for v in box] for box in value]
        return []

    candidate_boxes = boxes_to_list(
        result.get("candidate_boxes", result.get("pending_candidate_boxes"))
    )
    reported_boxes = boxes_to_list(result.get("reported_boxes", result.get("boxes")))

    cand_ids = result.get("candidate_track_ids") or []
    rep_ids = result.get("reported_track_ids") or []

    return {
        "detected": bool(result.get("detected", False)),
        "status": bool(result.get("status", False)),
        "candidate_boxes": candidate_boxes,
        "reported_boxes": reported_boxes,
        "candidate_track_ids": [int(x) for x in cand_ids],
        "reported_track_ids": [int(x) for x in rep_ids],
        "debug": result.get("debug", {}),
    }

@app.post("/reset_camera/{camera_id}")
def reset_camera(camera_id: str):
    processor: StorageViolationFrameProcessor = app.state.processor
    processor.reset_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id}


def _unlink_temp(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@app.get("/demo/process_data_video", summary="Демо: пары ideal+video из /app/data → MP4 с боксами")
async def demo_process_data_video(
    duration_sec: float = Query(
        10.0,
        ge=-1.0,
        description="Секунд с начала ролика; -1 = весь файл до конца",
    ),
    preset: int = Query(
        1,
        ge=1,
        le=4,
        description="1=ideal+video1, 2=ofis_big, 3=ofis_small, 4=ideal2+video2",
    ),
):
    """
    Тестовый эндпоинт: читает выбранную пару файлов из каталога демо (см. ``DATA_DIR`` в ``demo_pipeline.py``),
    прогоняет пайплайн без HTTP к самому себе, возвращает выходное MP4.
    """
    processor: StorageViolationFrameProcessor = app.state.processor
    ideal_storage: IdealImageStorage = app.state.ideal_storage

    try:
        out_path = await run_in_threadpool(
            run_demo_video,
            processor,
            ideal_storage,
            duration_sec,
            preset,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="demo_output.mp4",
        background=BackgroundTask(_unlink_temp, out_path),
    )


@app.get(
    "/demo/process_data_video_by_names",
    summary="Демо: свои имена ideal и video в каталоге демо → MP4",
)
async def demo_process_data_video_by_names(
    ideal_name: str = Query(
        ...,
        min_length=1,
        max_length=255,
        description="Имя файла ideal в каталоге демо (/app/data в Docker), напр. ofis_small.png",
    ),
    video_name: str = Query(
        ...,
        min_length=1,
        max_length=255,
        description="Имя файла видео в каталоге демо",
    ),
    duration_sec: float = Query(
        10.0,
        ge=-1.0,
        description="Секунд с начала; -1 = весь файл",
    ),
):
    """Файлы ищутся только в каталоге демо (см. ``DATA_DIR`` в ``demo_pipeline.py``), без подпапок."""
    processor: StorageViolationFrameProcessor = app.state.processor
    ideal_storage: IdealImageStorage = app.state.ideal_storage

    try:
        out_path = await run_in_threadpool(
            run_demo_video_named,
            processor,
            ideal_storage,
            duration_sec,
            ideal_name,
            video_name,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="demo_output.mp4",
        background=BackgroundTask(_unlink_temp, out_path),
    )