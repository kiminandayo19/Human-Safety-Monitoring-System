"""Persistence helpers for detection results."""

from pathlib import Path

from app.core.config import settings
from app.schemas.detection import DetectionType, PersonDetectionResult
from app.services.detector import SafetyDetector


def save_detection_result(
    detector: SafetyDetector,
    detection_type: DetectionType,
    image_bytes: bytes,
    filename: str | None,
    result: PersonDetectionResult,
) -> None:
    """Persist the annotated image and the JSON response to disk.

    Both files share the template ``{detection_type}_detection_{id}.{ext}``,
    the image using the original extension and the JSON using ``.json``.
    """
    output_dir = Path(settings.DETECTION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(filename or "").suffix.lstrip(".") or "jpg"
    stem = f"{detection_type.value}_detection_{result.detection_id}"

    annotated = detector.annotate(image_bytes, result.boxes, extension)
    (output_dir / f"{stem}.{extension}").write_bytes(annotated)
    (output_dir / f"{stem}.json").write_text(result.model_dump_json(indent=2))
