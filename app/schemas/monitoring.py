"""Pydantic schemas for safety monitoring events and alerts."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.detection import Detection


class Severity(str, Enum):
    """Severity level of a safety event."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SafetyEvent(BaseModel):
    """A safety-relevant event derived from one or more detections."""

    event_id: str
    camera_id: str
    severity: Severity
    message: str
    detections: list[Detection] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Alert(SafetyEvent):
    """An escalated safety event that requires operator attention."""

    acknowledged: bool = False


class HealthStatus(BaseModel):
    """Service health response."""

    status: str = "ok"
    version: str
    environment: str
