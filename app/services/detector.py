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
from app.schemas.tracking import (
    IDSwitchRate,
    TrackedPerson,
    TrackingData,
    TrackingFrame,
)

logger = get_logger(__name__)


class _IDSwitchTracker:
    """Measures tracking-ID consistency via the ID switch rate method.

    A switch is counted when the track id occupying an approximate spatial
    cell changes between observations — the same approach as the prototype in
    ``scripts/tracker.py``.
    """

    def __init__(self) -> None:
        self._history: dict[tuple, int] = {}
        self.id_switches: int = 0
        self.total_frames: int = 0

    @staticmethod
    def _cell(x1: float, y1: float, x2: float, y2: float) -> tuple:
        """Approximate spatial identity — 50px grid cell."""
        return (int(x1) // 50, int(y1) // 50, int(x2) // 50, int(y2) // 50)

    def observe(self, x1: float, y1: float, x2: float, y2: float,
                track_id: int) -> None:
        key = self._cell(x1, y1, x2, y2)
        if key in self._history and self._history[key] != track_id:
            self.id_switches += 1
        self._history[key] = track_id

    def tick(self) -> None:
        self.total_frames += 1

    def rate(self) -> IDSwitchRate:
        pct = (self.id_switches / self.total_frames * 100.0
               if self.total_frames else 0.0)
        return IDSwitchRate(
            id_switches=self.id_switches,
            total_frames=self.total_frames,
            rate=f"{pct:.2f}%",
        )

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

    @staticmethod
    def _ppe_status_for_person(
        person_box: tuple[float, float, float, float],
        ppe_items: list[tuple[SafetyLabel, tuple[float, float, float, float]]],
    ) -> tuple[bool, bool, bool]:
        """Derive (helmet, vest, mask) for a person from nearby PPE boxes.

        A PPE box is attributed to a person when its center falls inside the
        person's bounding box.
        """
        px1, py1, px2, py2 = person_box
        helmet = vest = mask = False
        for label, (bx1, by1, bx2, by2) in ppe_items:
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            if not (px1 <= cx <= px2 and py1 <= cy <= py2):
                continue
            if label is SafetyLabel.HARD_HAT:
                helmet = True
            elif label is SafetyLabel.SAFETY_VEST:
                vest = True
            elif label is SafetyLabel.MASK:
                mask = True
        return helmet, vest, mask

    def track_persons(
        self,
        video_path: str,
        output_video_path: str,
        conf: float = 0.3,
    ) -> TrackingData:
        """Track persons (+ their PPE) through a video and write an overlay.

        Uses ByteTrack on both the person and PPE models. For every tracked
        person the associated PPE state is drawn on the frame, the annotated
        video is written to ``output_video_path``, and an ID switch rate is
        accumulated for tracking-ID consistency.

        Args:
            video_path: Path to the source video.
            output_video_path: Path to write the annotated MP4 to.
            conf: Minimum confidence threshold for both models.

        Returns:
            :class:`TrackingData` with per-frame results and summary metrics.
        """
        if not self.is_ready():
            raise RuntimeError(
                "Detector models are not loaded. Call load() first.")

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video.")

        src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps,
            (width, height),
        )

        id_switch = _IDSwitchTracker()
        frames: list[TrackingFrame] = []
        total_time = 0.0
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                start = time.perf_counter()

                person_res = self._person_model.track(
                    frame, persist=True, classes=[0], conf=conf,
                    tracker="bytetrack.yaml", verbose=False)[0]
                ppe_res = self._ppe_model.track(
                    frame, persist=True, conf=conf,
                    tracker="bytetrack.yaml", verbose=False)[0]

                ppe_items: list[
                    tuple[SafetyLabel, tuple[float, float, float, float]]] = []
                if ppe_res.boxes is not None:
                    for box, cls_idx in zip(
                        ppe_res.boxes.xyxy, ppe_res.boxes.cls
                    ):
                        label = self._label_for(
                            DetectionType.PPE, int(cls_idx))
                        ppe_items.append(
                            (label, tuple(float(v) for v in box)))

                persons = self._track_persons_in_frame(
                    frame, person_res, ppe_items, id_switch)

                elapsed = time.perf_counter() - start
                total_time += elapsed
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                writer.write(frame)
                frames.append(
                    TrackingFrame(frame_index=frame_index, persons=persons))
                id_switch.tick()
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        avg_fps = frame_index / total_time if total_time > 0 else 0.0
        return TrackingData(
            fps=round(avg_fps, 2),
            total_frames=frame_index,
            id_switch_rate=id_switch.rate(),
            frames=frames,
        )

    def _track_persons_in_frame(
        self,
        frame: "np.ndarray",
        person_res: Any,
        ppe_items: list[tuple[SafetyLabel, tuple[float, float, float, float]]],
        id_switch: _IDSwitchTracker,
    ) -> list[TrackedPerson]:
        """Build tracked-person records for one frame and draw their overlays."""
        persons: list[TrackedPerson] = []
        if person_res.boxes is None or person_res.boxes.id is None:
            return persons

        for box, track_id in zip(person_res.boxes.xyxy, person_res.boxes.id):
            x1, y1, x2, y2 = (float(v) for v in box)
            person_id = str(int(track_id))
            helmet, vest, mask = self._ppe_status_for_person(
                (x1, y1, x2, y2), ppe_items)
            id_switch.observe(x1, y1, x2, y2, int(track_id))

            # Green when fully compliant, red otherwise.
            compliant = helmet and vest and mask
            color = (0, 255, 0) if compliant else (0, 0, 255)
            label = (f"ID{person_id} H:{int(helmet)} "
                     f"V:{int(vest)} M:{int(mask)}")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            cv2.putText(frame, label, (int(x1), max(int(y1) - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            persons.append(
                TrackedPerson(
                    person_id=person_id,
                    helmet=helmet,
                    vest=vest,
                    mask=mask,
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return persons


# Singleton instance shared across the application.
detector = SafetyDetector()
