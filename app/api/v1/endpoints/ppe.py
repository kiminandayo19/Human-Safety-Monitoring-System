"""PPE detection endpoints — run the PPE model over uploaded frames."""

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_detector
from app.core.config import settings
from app.schemas.detection import PersonDetectionResult
from app.services.detector import SafetyDetector

router = APIRouter(tags=["ppe"])


@router.post(
    "/detect-ppe",
    response_model=PersonDetectionResult,
    status_code=status.HTTP_200_OK,
)
async def detect_ppe(
    file: UploadFile = File(..., description="Image/frame to analyze"),
    save_detection: bool = Form(default=False),
    detector: SafetyDetector = Depends(get_detector),
) -> PersonDetectionResult:
    """Detect PPE compliance classes in a single uploaded image."""
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
        boxes = detector.detect_ppe(
            image_bytes, conf=settings.PERSON_DETECTION_THRESHOLD)
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
        output_dir = Path(settings.DETECTION_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        extension = Path(file.filename or "").suffix.lstrip(".") or "jpg"
        out_path = output_dir / f"ppe_detection_{result.detection_id}.{extension}"
        out_path.write_bytes(image_bytes)

    return result
