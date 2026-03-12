from __future__ import annotations

from typing import List, Optional

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

class RuntimeParamsRequest(BaseModel):
    threshold_hits: Optional[int] = None
    threshold_time_sec: Optional[float] = None
    min_side: Optional[int] = None


class RuntimeParamsResponse(BaseModel):
    threshold_hits: int
    threshold_time_sec: float
    min_side: int