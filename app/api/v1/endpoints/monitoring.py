"""Monitoring endpoints — evaluate detections into safety events and alerts."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_detector, get_monitor
from app.schemas.monitoring import Alert, SafetyEvent
from app.services.detector import SafetyDetector
from app.services.safety_monitor import SafetyMonitor

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/evaluate", response_model=list[SafetyEvent])
async def evaluate_frame(
    camera_id: str = Form(...),
    file: UploadFile = File(..., description="Image/frame to evaluate"),
    detector: SafetyDetector = Depends(get_detector),
    monitor: SafetyMonitor = Depends(get_monitor),
) -> list[SafetyEvent]:
    """Run detection on a frame and return derived safety events."""
    if not detector.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection model is not ready.",
        )

    image_bytes = await file.read()
    result = detector.predict(image_bytes)
    return monitor.evaluate(camera_id, result)


@router.post("/alerts", response_model=list[Alert])
async def evaluate_alerts(
    camera_id: str = Form(...),
    file: UploadFile = File(...),
    detector: SafetyDetector = Depends(get_detector),
    monitor: SafetyMonitor = Depends(get_monitor),
) -> list[Alert]:
    """Run detection and return only events escalated to operator alerts."""
    image_bytes = await file.read()
    result = detector.predict(image_bytes)
    events = monitor.evaluate(camera_id, result)
    return monitor.to_alerts(events)
