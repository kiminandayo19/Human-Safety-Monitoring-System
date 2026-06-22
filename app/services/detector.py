"""Computer-vision detector service.

This is a boilerplate placeholder. Swap :class:`SafetyDetector` internals with
a real model (e.g. YOLO / Ultralytics, Detectron2, ONNX Runtime, OpenVINO).
The rest of the application only depends on the public interface and the
``DetectionResult`` schema, so the model can be replaced without touching the
API layer.
"""

import time

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.detection import DetectionResult

logger = get_logger(__name__)


class SafetyDetector:
    """Wraps a computer-vision model used for human-safety detection."""

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.model_path = model_path or settings.MODEL_PATH
        self.device = device or settings.DEVICE
        self._model = None  # Loaded lazily in ``load()``.

    def load(self) -> None:
        """Load model weights into memory.

        Replace this stub with real model loading, e.g.::

            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        """
        logger.info("Loading detection model from %s on %s", self.model_path, self.device)
        # self._model = <load your model here>
        self._model = object()  # placeholder so ``is_ready`` returns True

    def is_ready(self) -> bool:
        """Return True once the model is loaded and ready for inference."""
        return self._model is not None

    def predict(self, image_bytes: bytes, frame_id: str | None = None) -> DetectionResult:
        """Run inference on a single image/frame.

        Args:
            image_bytes: Raw image bytes (e.g. JPEG/PNG).
            frame_id: Optional identifier for the frame.

        Returns:
            A :class:`DetectionResult` with detected safety events.
        """
        if not self.is_ready():
            raise RuntimeError("Detector model is not loaded. Call load() first.")

        start = time.perf_counter()

        # --- Replace with real preprocessing + inference + postprocessing ---
        # image = decode(image_bytes)
        # raw = self._model(image, conf=settings.DETECTION_CONFIDENCE_THRESHOLD)
        # detections = postprocess(raw)
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
