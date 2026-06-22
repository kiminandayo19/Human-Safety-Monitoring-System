"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends

from app.api.deps import get_detector
from app.core.config import settings
from app.schemas.monitoring import HealthStatus
from app.services.detector import SafetyDetector

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    """Liveness probe."""
    return HealthStatus(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get("/ready")
def ready(detector: SafetyDetector = Depends(get_detector)) -> dict:
    """Readiness probe — reports whether the detection model is loaded."""
    return {"ready": detector.is_ready()}
