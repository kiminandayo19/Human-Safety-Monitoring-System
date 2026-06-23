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
    DetectionType,
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

    def detect(
        self,
        detection_type: DetectionType,
        image_bytes: bytes,
        conf: float = 0.3,
    ) -> list[Detection]:
        """Run detection on a single image with the requested model.

        Args:
            detection_type: Which model to run (``ppe`` or ``person``).
            image_bytes: Raw image bytes (e.g. JPEG/PNG).
            conf: Minimum confidence threshold for returned boxes.

        Returns:
            One :class:`Detection` per detected object/violation.
        """
        if not self.is_ready():
            raise RuntimeError(
                "Detector models are not loaded. Call load() first.")

        image = self._decode_image(image_bytes)

        if detection_type is DetectionType.PERSON:
            # class 0 == "person" in the pretrained YOLOv11 model.
            results = self._person_model(
                image, classes=[0], conf=conf, verbose=False)
        else:
            results = self._ppe_model(image, conf=conf, verbose=False)

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for xyxy, cls_idx, confidence in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            ):
                detections.append(
                    Detection(
                        id=str(uuid.uuid4()),
                        label=self._label_for(detection_type, int(cls_idx)),
                        confidence=float(confidence),
                        box=BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                    )
                )
        return detections

    @staticmethod
    def _label_for(detection_type: DetectionType, cls_idx: int) -> SafetyLabel:
        """Resolve a model class index to a :class:`SafetyLabel`."""
        if detection_type is DetectionType.PERSON:
            return SafetyLabel.PERSON
        if 0 <= cls_idx < len(PPE_CLASS_LABELS):
            return PPE_CLASS_LABELS[cls_idx]
        return SafetyLabel.UNKNOWN

    def annotate(
        self,
        image_bytes: bytes,
        detections: list[Detection],
        extension: str = "jpg",
    ) -> bytes:
        """Draw detection boxes/labels onto the image and re-encode it.

        Args:
            image_bytes: The original image bytes.
            detections: Boxes to overlay.
            extension: Output image extension (e.g. ``jpg``, ``png``).

        Returns:
            The annotated image encoded as bytes.
        """
        image = self._decode_image(image_bytes)

        for det in detections:
            x1, y1 = int(det.box.x1), int(det.box.y1)
            x2, y2 = int(det.box.x2), int(det.box.y2)
            # Violations (the "no_*" classes) are drawn red, otherwise green.
            is_violation = det.label.value.startswith("no_")
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            label = f"{det.label.value} {det.confidence:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        ext = extension if extension.startswith(".") else f".{extension}"
        ok, buffer = cv2.imencode(ext, image)
        if not ok:
            # Fall back to JPEG for extensions OpenCV cannot encode.
            ok, buffer = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError("Could not encode the annotated image.")
        return buffer.tobytes()

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
