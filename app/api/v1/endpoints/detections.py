"""Detection endpoints — run the CV model over uploaded frames."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_detector
from app.schemas.detection import DetectionResult
from app.services.detector import SafetyDetector

router = APIRouter(prefix="/detections", tags=["detections"])


@router.post("", response_model=DetectionResult, status_code=status.HTTP_200_OK)
async def detect(
    file: UploadFile = File(..., description="Image/frame to analyze"),
    frame_id: str | None = Form(default=None),
    detector: SafetyDetector = Depends(get_detector),
) -> DetectionResult:
    """Run safety detection on a single uploaded image/frame."""
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

    return detector.predict(image_bytes, frame_id=frame_id)
