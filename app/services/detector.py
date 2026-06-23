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
import uuid
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.detection import (
    BoundingBox,
    Detection,
    DetectionResult,
    SafetyLabel,
)

logger = get_logger(__name__)

# Maps PPE model class indices to safety labels — order matches CLASS_NAMES
# in scripts/tracker.py.
PPE_CLASS_LABELS: tuple[SafetyLabel, ...] = (
    SafetyLabel.HARD_HAT,        # 0 - Hard Hat
    SafetyLabel.NO_HARD_HAT,     # 1 - No Hard Hat
    SafetyLabel.MASK,            # 2 - Mask
    SafetyLabel.NO_MASK,         # 3 - No Mask
    SafetyLabel.SAFETY_VEST,     # 4 - Safety Vest
    SafetyLabel.NO_SAFETY_VEST,  # 5 - No Safety Vest
)


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

    @staticmethod
    def _decode_image(image_bytes: bytes) -> "np.ndarray":
        """Decode raw image bytes into a BGR numpy array."""
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the uploaded image.")
        return image

    def detect_persons(
        self, image_bytes: bytes, conf: float = 0.3
    ) -> list[Detection]:
        """Detect persons in a single image using the person model.

        Args:
            image_bytes: Raw image bytes (e.g. JPEG/PNG).
            conf: Minimum confidence threshold for returned boxes.

        Returns:
            One :class:`Detection` per detected person.
        """
        if self._person_model is None:
            raise RuntimeError(
                "Person model is not loaded. Call load() first.")

        image = self._decode_image(image_bytes)

        # class 0 == "person" in the pretrained YOLOv11 model.
        results = self._person_model(
            image, classes=[0], conf=conf, verbose=False)

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for xyxy, confidence in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                detections.append(
                    Detection(
                        id=str(uuid.uuid4()),
                        label=SafetyLabel.PERSON,
                        confidence=float(confidence),
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return detections

    def detect_ppe(
        self, image_bytes: bytes, conf: float = 0.3
    ) -> list[Detection]:
        """Detect PPE compliance classes in a single image using the PPE model.

        Args:
            image_bytes: Raw image bytes (e.g. JPEG/PNG).
            conf: Minimum confidence threshold for returned boxes.

        Returns:
            One :class:`Detection` per detected PPE item/violation.
        """
        if self._ppe_model is None:
            raise RuntimeError(
                "PPE model is not loaded. Call load() first.")

        image = self._decode_image(image_bytes)

        results = self._ppe_model(image, conf=conf, verbose=False)

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for xyxy, cls_idx, confidence in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            ):
                idx = int(cls_idx)
                label = (
                    PPE_CLASS_LABELS[idx]
                    if 0 <= idx < len(PPE_CLASS_LABELS)
                    else SafetyLabel.UNKNOWN
                )
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                detections.append(
                    Detection(
                        id=str(uuid.uuid4()),
                        label=label,
                        confidence=float(confidence),
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return detections

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
