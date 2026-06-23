"""Unified detection endpoint — run the PPE or person model over a frame."""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_detector
from app.core.config import settings
from app.schemas.detection import DetectionType, PersonDetectionResult
from app.services.detector import SafetyDetector
from app.utils.storage import save_detection_result

router = APIRouter(tags=["detection"])


@router.post(
    "/detect",
    response_model=PersonDetectionResult,
    status_code=status.HTTP_200_OK,
)
async def detect(
    file: UploadFile = File(..., description="Image/frame to analyze"),
    detection_type: DetectionType = Form(
        ..., description="Which model to run: ppe | person"),
    save_detection: bool = Form(default=False),
    detector: SafetyDetector = Depends(get_detector),
) -> PersonDetectionResult:
    """Detect PPE compliance or persons in a single uploaded image."""
    if not detector.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection model is not ready.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image upload.",
        )

    try:
        boxes = detector.detect(
            detection_type,
            image_bytes,
            conf=settings.PERSON_DETECTION_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    result = PersonDetectionResult(
        detection_id=str(uuid.uuid4()),
        input_name=file.filename or "unknown",
        boxes=boxes,
    )

    if save_detection:
        save_detection_result(
            detector, detection_type, image_bytes, file.filename, result)

    return result
