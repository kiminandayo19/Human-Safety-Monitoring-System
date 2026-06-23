"""Person tracking endpoint — track persons + PPE through an uploaded video."""

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_detector
from app.core.config import settings
from app.schemas.tracking import PersonTrackingResult
from app.services.detector import SafetyDetector
from app.utils.storage import save_tracking_json, tracking_output_paths

router = APIRouter(tags=["tracking"])


@router.post(
    "/track-person",
    response_model=PersonTrackingResult,
    status_code=status.HTTP_200_OK,
)
async def track_person(
    file: UploadFile = File(..., description="Video to track"),
    detector: SafetyDetector = Depends(get_detector),
) -> PersonTrackingResult:
    """Track persons and their PPE through a video, returning an overlay video
    and a per-frame tracking JSON file."""
    if not detector.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection model is not ready.",
        )

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty video upload.",
        )

    tracking_id = str(uuid.uuid4())
    video_path, json_path = tracking_output_paths(tracking_id)

    suffix = Path(file.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        data = detector.track_persons(
            tmp_path,
            str(video_path),
            conf=settings.PERSON_DETECTION_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    save_tracking_json(json_path, tracking_id, file.filename or "unknown", data)

    return PersonTrackingResult(
        tracking_id=tracking_id,
        input_name=file.filename or "unknown",
        fps=data.fps,
        total_frames=data.total_frames,
        id_switch_rate=data.id_switch_rate,
        video_path=str(video_path),
        json_path=str(json_path),
    )
