from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .demo_pipeline import run_demo_video, run_demo_video_named, sanitize_data_filename
from .ideal_storage import IdealImageStorage
from .processor import StorageViolationFrameProcessor
from .runner_common import (
    apply_model_config_updates,
    apply_processor_params_from_request,
    build_processor,
    build_runtime_params_response,
    draw_result,
    encode_image_bgr_jpeg_b64,
    pack_output_zip,
    parse_polygons_json,
    prepare_ideal_bgr,
    run_on_image,
    run_video_pipeline,
    safe_unlink,
    safe_unlink_many,
)
from .schemas import RuntimeParamsRequest, RuntimeParamsResponse

LOGGER = logging.getLogger("storage_violation_api")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)


MODEL_WEIGHTS_PATH = "weights/cd_weights.pt"
MODEL_DEVICE = "auto"
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
    import cv2

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
    try:
        return parse_polygons_json(polygons_json)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _save_upload_to_temp(upload: UploadFile, suffix: str) -> str:
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _video_file_response(
    video_path: str,
    jsonl_path: Optional[str],
    output_name: str,
    include_jsonl: bool,
) -> FileResponse:
    try:
        download_name = sanitize_data_filename(output_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if include_jsonl and jsonl_path:
        zip_path = pack_output_zip(video_path, jsonl_path, download_name)
        zip_name = (
            download_name.replace(".mp4", ".zip")
            if download_name.endswith(".mp4")
            else f"{download_name}.zip"
        )
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_name,
            background=BackgroundTask(safe_unlink_many, video_path, jsonl_path, zip_path),
        )

    if not download_name.endswith(".mp4"):
        download_name = f"{download_name}.mp4"

    cleanup = [video_path]
    if jsonl_path:
        cleanup.append(jsonl_path)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=download_name,
        background=BackgroundTask(safe_unlink_many, *cleanup),
    )


async def _run_media_pipeline(
    processor: StorageViolationFrameProcessor,
    ideal_storage: IdealImageStorage,
    camera_id: str,
    ideal_override: Optional[np.ndarray],
    polygons_np: Optional[list[np.ndarray]],
    *,
    video_path: Optional[str] = None,
    rtsp_url: Optional[str] = None,
    every_n_frames: int = 1,
    max_frames: Optional[int] = None,
    duration_sec: Optional[float] = None,
    rtsp_reconnect_sec: float = 1.0,
    include_jsonl: bool = False,
    output_name: str = "output.mp4",
    temp_video_to_cleanup: Optional[str] = None,
) -> FileResponse:
    try:
        ideal_bgr = prepare_ideal_bgr(processor, ideal_storage, camera_id, ideal_override)
    except ValueError as e:
        if temp_video_to_cleanup:
            safe_unlink(temp_video_to_cleanup)
        raise HTTPException(status_code=404, detail=str(e)) from e

    try:
        out_video, jsonl_path = await run_in_threadpool(
            run_video_pipeline,
            processor,
            camera_id,
            ideal_bgr,
            video_path=video_path,
            rtsp_url=rtsp_url,
            polygons=polygons_np,
            every_n_frames=max(1, int(every_n_frames)),
            reconnect_sec=float(rtsp_reconnect_sec),
            max_frames=max_frames,
            duration_sec=duration_sec,
            include_jsonl=bool(include_jsonl),
        )
    except ValueError as e:
        if temp_video_to_cleanup:
            safe_unlink(temp_video_to_cleanup)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        if temp_video_to_cleanup:
            safe_unlink(temp_video_to_cleanup)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if temp_video_to_cleanup:
            safe_unlink(temp_video_to_cleanup)

    return _video_file_response(out_video, jsonl_path, output_name, include_jsonl)


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("[API] Loading model and processor...")

    processor = build_processor(
        weights_path=MODEL_WEIGHTS_PATH,
        device=MODEL_DEVICE,
        half=MODEL_HALF,
        threshold=MODEL_THRESHOLD,
        inp_ch=MODEL_INP_CH,
        logger=LOGGER,
        **PROCESSOR_KWARGS,
    )
    ideal_storage = IdealImageStorage()

    app.state.processor = processor
    app.state.ideal_storage = ideal_storage
    app.state.model_config = {
        "weights_path": MODEL_WEIGHTS_PATH,
        "device": MODEL_DEVICE,
        "threshold": MODEL_THRESHOLD,
        "half": MODEL_HALF,
        "inp_ch": MODEL_INP_CH,
    }

    LOGGER.info("[API] Processor is ready")
    yield

    LOGGER.info("[API] Shutting down processor...")
    processor.close()
    ideal_storage.clear()
    LOGGER.info("[API] Shutdown complete")


app = FastAPI(
    title="Storage Violation API",
    version="3.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ideal/{camera_id}")
async def upload_ideal(camera_id: str, image: UploadFile = File(...)):
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
    return {"camera_id": camera_id, "exists": ideal_storage.exists(camera_id)}


@app.get("/ideal")
def list_ideal_cameras():
    ideal_storage: IdealImageStorage = app.state.ideal_storage
    camera_ids = ideal_storage.list_camera_ids()
    return {"camera_ids": camera_ids, "count": len(camera_ids)}


@app.delete("/ideal/{camera_id}")
def delete_ideal(camera_id: str):
    ideal_storage: IdealImageStorage = app.state.ideal_storage
    deleted = ideal_storage.delete(camera_id)
    return {"camera_id": camera_id, "deleted": deleted}


@app.get("/params", response_model=RuntimeParamsResponse)
def get_params():
    processor: StorageViolationFrameProcessor = app.state.processor
    return build_runtime_params_response(processor, app.state.model_config)


@app.post("/params", response_model=RuntimeParamsResponse)
def set_params(body: RuntimeParamsRequest):
    processor: StorageViolationFrameProcessor = app.state.processor
    model_cfg: dict[str, Any] = app.state.model_config

    try:
        model_cfg = apply_model_config_updates(processor, model_cfg, body, logger=LOGGER)
        apply_processor_params_from_request(processor, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    app.state.model_config = model_cfg
    return build_runtime_params_response(processor, model_cfg)


@app.post("/process_frame")
async def process_frame(
    camera_id: str = Form(...),
    frame: UploadFile = File(...),
    polygons: Optional[str] = Form(None),
    ideal_image: Optional[UploadFile] = File(None),
    return_annotated_image: bool = Form(False),
):
    processor: StorageViolationFrameProcessor = app.state.processor
    ideal_storage: IdealImageStorage = app.state.ideal_storage

    ideal_override = await read_uploaded_image(ideal_image) if ideal_image is not None else None
    frame_bgr = await read_uploaded_image(frame)
    polygons_np = normalize_polygons_from_json(polygons)

    try:
        payload = await run_in_threadpool(
            run_on_image,
            processor,
            ideal_storage,
            camera_id,
            frame_bgr,
            ideal_override,
            polygons_np,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if return_annotated_image:
        vis = draw_result(frame_bgr, payload)
        payload["annotated_image_b64"] = encode_image_bgr_jpeg_b64(vis)
    return payload


@app.post("/process_video", summary="Прогон загруженного видео → MP4 с боксами")
async def process_video(
    camera_id: str = Form(...),
    video: UploadFile = File(...),
    ideal_image: Optional[UploadFile] = File(None),
    polygons: Optional[str] = Form(None),
    every_n_frames: int = Form(1),
    max_frames: Optional[int] = Form(None),
    duration_sec: Optional[float] = Form(None),
    output_name: str = Form("output.mp4"),
    include_jsonl: bool = Form(False),
):
    processor: StorageViolationFrameProcessor = app.state.processor
    ideal_storage: IdealImageStorage = app.state.ideal_storage

    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    video_path = await _save_upload_to_temp(video, suffix=suffix)
    ideal_override = await read_uploaded_image(ideal_image) if ideal_image is not None else None
    polygons_np = normalize_polygons_from_json(polygons)

    return await _run_media_pipeline(
        processor,
        ideal_storage,
        camera_id,
        ideal_override,
        polygons_np,
        video_path=video_path,
        every_n_frames=every_n_frames,
        max_frames=max_frames,
        duration_sec=duration_sec,
        include_jsonl=include_jsonl,
        output_name=output_name,
        temp_video_to_cleanup=video_path,
    )


@app.post("/process_rtsp", summary="Прогон RTSP-потока → MP4 с боксами")
async def process_rtsp(
    camera_id: str = Form(...),
    rtsp_url: str = Form(...),
    ideal_image: Optional[UploadFile] = File(None),
    polygons: Optional[str] = Form(None),
    every_n_frames: int = Form(1),
    max_frames: Optional[int] = Form(None),
    duration_sec: Optional[float] = Form(None),
    rtsp_reconnect_sec: float = Form(1.0),
    output_name: str = Form("output.mp4"),
    include_jsonl: bool = Form(False),
):
    processor: StorageViolationFrameProcessor = app.state.processor
    ideal_storage: IdealImageStorage = app.state.ideal_storage

    ideal_override = await read_uploaded_image(ideal_image) if ideal_image is not None else None
    polygons_np = normalize_polygons_from_json(polygons)

    return await _run_media_pipeline(
        processor,
        ideal_storage,
        camera_id,
        ideal_override,
        polygons_np,
        rtsp_url=rtsp_url,
        every_n_frames=every_n_frames,
        max_frames=max_frames,
        duration_sec=duration_sec,
        rtsp_reconnect_sec=rtsp_reconnect_sec,
        include_jsonl=include_jsonl,
        output_name=output_name,
    )


@app.post("/reset_camera/{camera_id}")
def reset_camera(camera_id: str):
    processor: StorageViolationFrameProcessor = app.state.processor
    processor.reset_camera(camera_id)
    return {"status": "ok", "camera_id": camera_id}


@app.get("/demo/process_data_video", summary="Демо: пары ideal+video из /app/data → MP4 с боксами")
async def demo_process_data_video(
    duration_sec: float = Query(10.0, ge=-1.0),
    preset: int = Query(1, ge=1, le=4),
    output_name: str = Query("demo_output.mp4", min_length=1, max_length=255),
):
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

    try:
        download_name = sanitize_data_filename(output_name)
    except ValueError as e:
        safe_unlink(out_path)
        raise HTTPException(status_code=400, detail=str(e)) from e

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=download_name,
        background=BackgroundTask(safe_unlink, out_path),
    )


@app.get("/demo/process_data_video_by_names", summary="Демо: свои имена ideal и video → MP4")
async def demo_process_data_video_by_names(
    ideal_name: str = Query(..., min_length=1, max_length=255),
    video_name: str = Query(..., min_length=1, max_length=255),
    duration_sec: float = Query(10.0, ge=-1.0),
    output_name: str = Query("demo_output.mp4", min_length=1, max_length=255),
):
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

    try:
        download_name = sanitize_data_filename(output_name)
    except ValueError as e:
        safe_unlink(out_path)
        raise HTTPException(status_code=400, detail=str(e)) from e

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=download_name,
        background=BackgroundTask(safe_unlink, out_path),
    )
