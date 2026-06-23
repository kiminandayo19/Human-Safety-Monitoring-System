"""Pydantic schemas for detection requests and results."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field
from typing import List


class SafetyLabel(str, Enum):
    """Classes the safety detector can emit."""

    # PPE detection classes — match CLASS_NAMES in scripts/tracker.py.
    HARD_HAT = "hard_hat"
    NO_HARD_HAT = "no_hard_hat"
    MASK = "mask"
    NO_MASK = "no_mask"
    SAFETY_VEST = "safety_vest"
    NO_SAFETY_VEST = "no_safety_vest"

    PERSON = "person"
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

    id: str = Field(..., description="detection uuid")
    label: SafetyLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class DetectionResult(BaseModel):
    """Result of running the detector over a single frame/image."""

    frame_id: str | None = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    detections: list[Detection] = Field(default_factory=list)
    processing_ms: float = Field(...,
                                 description="Inference time in milliseconds")


class PersonDetectionResult(BaseModel):
    """Result of running the person detector over a single frame/image."""
    detection_id: str = Field(..., description="detection result uuid")
    input_name: str = Field(..., description="input filename")
    boxes: List[Detection]
