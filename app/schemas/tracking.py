"""Pydantic schemas for person tracking requests and results."""

from pydantic import BaseModel, Field

from app.schemas.detection import BoundingBox


class TrackedPerson(BaseModel):
    """A single tracked person and their PPE compliance in a frame."""

    person_id: str = Field(..., description="ByteTrack track id")
    helmet: bool = Field(..., description="Wearing a hard hat")
    vest: bool = Field(..., description="Wearing a safety vest")
    mask: bool = Field(..., description="Wearing a mask")
    box: BoundingBox


class TrackingFrame(BaseModel):
    """All tracked persons within a single video frame."""

    frame_index: int
    persons: list[TrackedPerson] = Field(default_factory=list)


class IDSwitchRate(BaseModel):
    """Tracking-ID consistency measured via the ID switch rate method."""

    id_switches: int
    total_frames: int
    rate: str = Field(..., description="ID switches as a percentage of frames")


class TrackingData(BaseModel):
    """Full per-frame tracking output (written to the JSON file)."""

    fps: float = Field(..., description="Average processing FPS")
    total_frames: int
    id_switch_rate: IDSwitchRate
    frames: list[TrackingFrame] = Field(default_factory=list)


class PersonTrackingResult(BaseModel):
    """API response for ``POST /track-person``."""

    tracking_id: str = Field(..., description="tracking result uuid")
    input_name: str = Field(..., description="input filename")
    fps: float
    total_frames: int
    id_switch_rate: IDSwitchRate
    video_path: str = Field(..., description="Path to the annotated video")
    json_path: str = Field(..., description="Path to the tracking JSON file")
