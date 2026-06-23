"""Persistence helpers for detection results."""

import json
from pathlib import Path

from app.core.config import settings
from app.schemas.detection import DetectionType, PersonDetectionResult
from app.schemas.tracking import TrackingData
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


def tracking_output_paths(tracking_id: str) -> tuple[Path, Path]:
    """Return ``(video_path, json_path)`` for a tracking run.

    Both share the template ``track_person_{id}`` — the annotated video uses
    ``.mp4`` and the tracking data uses ``.json``. The output directory is
    created if needed so the tracker can write the video straight to it.
    """
    output_dir = Path(settings.DETECTION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"track_person_{tracking_id}"
    return output_dir / f"{stem}.mp4", output_dir / f"{stem}.json"


def save_tracking_json(
    json_path: Path,
    tracking_id: str,
    input_name: str,
    data: TrackingData,
) -> None:
    """Write the full per-frame tracking data to a JSON file."""
    payload = {
        "tracking_id": tracking_id,
        "input_name": input_name,
        **data.model_dump(),
    }
    json_path.write_text(json.dumps(payload, indent=2))
