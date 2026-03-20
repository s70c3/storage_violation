from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .ideal_storage import IdealImageStorage
from .processor import StorageViolationFrameProcessor
from .segmentator import UNetVGG16Segmentator
from .schemas import RuntimeParamsRequest, RuntimeParamsResponse

LOGGER = logging.getLogger("storage_violation_api")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)


MODEL_WEIGHTS_PATH = "weights/cd_weights.pt"
MODEL_DEVICE = "cuda:0"   # e.g. "cuda:0"
MODEL_THRESHOLD = 0.5
MODEL_HALF = True
MODEL_INP_CH = 9

PROCESSOR_KWARGS = dict(
    threshold_hits=10,
    threshold_time_sec=0,
    min_side=10,
    delta = 0
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

    boxes_np = result.get("boxes", np.empty((0, 4), dtype=np.int32))
    boxes = boxes_np.astype(int).tolist() if len(boxes_np) > 0 else []

    return {
        "detected": bool(result["detected"]),
        "status": bool(result["status"]),
        "boxes": boxes,
        "debug": result.get("debug", {}),
    }


@app.post("/reset_camera/{camera_id}")
def reset_camera(camera_id: str):
    processor: StorageViolationFrameProcessor = app.state.processor
    processor.reset_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id}

@app.get("/params", response_model=RuntimeParamsResponse)
def get_params():
    processor: StorageViolationFrameProcessor = app.state.processor

    return RuntimeParamsResponse(
        threshold_hits=processor.threshold_hits,
        threshold_time_sec=processor.threshold_time_sec,
        min_side=processor.min_side,
    )

@app.post("/params", response_model=RuntimeParamsResponse)
def update_params(payload: RuntimeParamsRequest):

    processor: StorageViolationFrameProcessor = app.state.processor

    processor.update_runtime_params(
        threshold_hits=payload.threshold_hits,
        threshold_time_sec=payload.threshold_time_sec,
        min_side=payload.min_side,
    )

    return RuntimeParamsResponse(
        threshold_hits=processor.threshold_hits,
        threshold_time_sec=processor.threshold_time_sec,
        min_side=processor.min_side,
    )