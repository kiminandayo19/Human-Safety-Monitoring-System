"""Computer-vision detector service.

Wraps the two YOLO models the system relies on:

- **PPE detection** (``settings.PPE_MODEL_PATH``) — helmet / vest / mask
  compliance.
- **Person detection + tracking** (``settings.PERSON_MODEL_PATH``) — pretrained
  YOLOv11 used for person detection and multi-object tracking.

Both models are loaded once at application startup (see ``app.main.lifespan``).
The rest of the application only depends on the public interface
(:meth:`SafetyDetector.load`, :meth:`is_ready`, :meth:`predict`) and the
``DetectionResult`` schema, so the models can be swapped without touching the
API layer.
"""

import time
from typing import Any
from ultralytics import YOLO

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.detection import DetectionResult

logger = get_logger(__name__)


class SafetyDetector:
    """Wraps the computer-vision models used for human-safety detection."""

    def __init__(
        self,
        ppe_model_path: str | None = None,
        person_model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.ppe_model_path = ppe_model_path or settings.PPE_MODEL_PATH
        self.person_model_path = person_model_path or settings.PERSON_MODEL_PATH
        self.device = device or settings.DEVICE
        self._ppe_model: Any = None  # Loaded in ``load()``.
        self._person_model: Any = None  # Loaded in ``load()``.

    def load(self) -> None:
        """Load both model weights into memory.

        Called once at boot from the application lifespan. Imports
        ``ultralytics`` lazily so the dependency is only required when the
        service actually starts loading models.
        """
        logger.info("Loading PPE detection model from %s on %s",
                    self.ppe_model_path, self.device)
        self._ppe_model = YOLO(self.ppe_model_path)

        logger.info("Loading person tracking model from %s on %s",
                    self.person_model_path, self.device)
        self._person_model = YOLO(self.person_model_path)

        logger.info("Detection models loaded successfully on %s", self.device)

    def is_ready(self) -> bool:
        """Return True once both models are loaded and ready for inference."""
        return self._ppe_model is not None and self._person_model is not None

    def predict(self, image_bytes: bytes, frame_id: str | None = None) -> DetectionResult:
        """Run inference on a single image/frame.

        Args:
            image_bytes: Raw image bytes (e.g. JPEG/PNG).
            frame_id: Optional identifier for the frame.

        Returns:
            A :class:`DetectionResult` with detected safety events.
        """
        if not self.is_ready():
            raise RuntimeError(
                "Detector models are not loaded. Call load() first.")

        start = time.perf_counter()

        # --- Replace with real preprocessing + inference + postprocessing ---
        # image = decode(image_bytes)
        # ppe_raw = self._ppe_model(image, conf=settings.DETECTION_CONFIDENCE_THRESHOLD)
        # person_raw = self._person_model(image, classes=[0])
        # detections = postprocess(ppe_raw, person_raw)
        detections: list = []
        # -------------------------------------------------------------------

        processing_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(
            frame_id=frame_id,
            detections=detections,
            processing_ms=processing_ms,
        )


# Singleton instance shared across the application.
detector = SafetyDetector()
