"""Pydantic schemas for detection requests and results."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class SafetyLabel(str, Enum):
    """Classes the safety detector can emit."""

    PERSON = "person"
    NO_HELMET = "no_helmet"
    NO_VEST = "no_vest"
    FALL = "fall"
    RESTRICTED_ZONE = "restricted_zone_intrusion"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float = Field(..., description="Left edge")
    y1: float = Field(..., description="Top edge")
    x2: float = Field(..., description="Right edge")
    y2: float = Field(..., description="Bottom edge")


class Detection(BaseModel):
    """A single detected object/event in a frame."""

    label: SafetyLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class DetectionResult(BaseModel):
    """Result of running the detector over a single frame/image."""

    frame_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detections: list[Detection] = Field(default_factory=list)
    processing_ms: float = Field(..., description="Inference time in milliseconds")
