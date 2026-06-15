from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class UploadIdealRequest(BaseModel):
    image_b64: str = Field(..., description="Ideal image in base64-encoded image format")


class UploadIdealResponse(BaseModel):
    camera_id: str
    stored: bool
    path: Optional[str] = None


class ProcessFrameRequest(BaseModel):
    camera_id: str = Field(..., description="Unique camera id")
    frame_b64: str = Field(..., description="Current frame in base64-encoded image format")

    polygons: Optional[List[List[List[int]]]] = Field(
        default=None,
        description="List of polygons; each polygon is [[x,y], [x,y], ...]",
    )


class ProcessFrameResponse(BaseModel):
    detected: bool
    status: bool
    boxes: List[List[int]]
    debug: dict


class ModelParams(BaseModel):
    weights_path: str = "weights/cd_weights.pt"
    device: str = "auto"
    threshold: float = 0.5
    half: bool = False
    inp_ch: int = 9


class RuntimeParamsRequest(BaseModel):
    # processor
    min_side: Optional[int] = None
    threshold_time_sec: Optional[float] = Field(
        default=None,
        description="Alias for stationary_time_sec",
    )
    stationary_time_sec: Optional[float] = None
    max_long_side: Optional[int] = None
    ideal_mode: Optional[Literal["static", "first_frame", "median", "mean"]] = None
    ideal_frames: Optional[int] = Field(default=None, ge=1)
    bg_median_window: Optional[int] = Field(default=None, ge=1, description="Alias for ideal_frames")
    bg_median_update_every: Optional[int] = Field(default=None, ge=1)
    thr: Optional[float] = None
    morph_ksize: Optional[int] = None
    ema_tau_sec: Optional[float] = None
    ema_min_alpha: Optional[float] = None
    ema_max_alpha: Optional[float] = None
    recent_update_every: Optional[int] = Field(default=None, ge=1)
    visualization: Optional[bool] = None
    tracker_track_activation_threshold: Optional[float] = None
    tracker_lost_track_buffer: Optional[int] = None
    tracker_minimum_matching_threshold: Optional[float] = None
    tracker_frame_rate: Optional[int] = None
    tracker_minimum_consecutive_frames: Optional[int] = None
    tracker_max_center_shift_px: Optional[float] = None
    tracker_max_center_shift_norm: Optional[float] = None
    # model (reload if device/weights/half/inp_ch change; threshold updates in-place)
    weights_path: Optional[str] = None
    device: Optional[str] = None
    threshold: Optional[float] = None
    half: Optional[bool] = None
    inp_ch: Optional[int] = None


class RuntimeParamsResponse(BaseModel):
    min_side: int
    stationary_time_sec: float
    max_long_side: int
    ideal_mode: Literal["static", "first_frame", "median", "mean"]
    ideal_frames: int
    bg_median_window: int
    bg_median_update_every: int
    thr: float
    morph_ksize: int
    ema_tau_sec: float
    ema_min_alpha: float
    ema_max_alpha: float
    recent_update_every: int
    visualization: bool
    tracker_track_activation_threshold: float
    tracker_lost_track_buffer: int
    tracker_minimum_matching_threshold: float
    tracker_frame_rate: int
    tracker_minimum_consecutive_frames: int
    tracker_max_center_shift_px: float
    tracker_max_center_shift_norm: float
    model: ModelParams
