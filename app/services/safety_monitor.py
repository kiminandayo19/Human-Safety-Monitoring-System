"""Safety monitoring business logic.

Turns raw detections into safety events and escalates them to alerts based on
configurable rules. This is boilerplate — extend ``evaluate`` with your own
rules (zone intrusion, missing PPE, fall detection, dwell time, etc.).
"""

import uuid

from app.core.config import settings
from app.schemas.detection import DetectionResult, SafetyLabel
from app.schemas.monitoring import Alert, SafetyEvent, Severity

# Detection labels that should always be treated as critical safety events.
_CRITICAL_LABELS = {SafetyLabel.FALL, SafetyLabel.RESTRICTED_ZONE}
_WARNING_LABELS = {
    SafetyLabel.NO_HARD_HAT,
    SafetyLabel.NO_MASK,
    SafetyLabel.NO_SAFETY_VEST,
}


def _severity_for(label: SafetyLabel) -> Severity:
    if label in _CRITICAL_LABELS:
        return Severity.CRITICAL
    if label in _WARNING_LABELS:
        return Severity.WARNING
    return Severity.INFO


class SafetyMonitor:
    """Evaluates detection results against safety rules."""

    def evaluate(self, camera_id: str, result: DetectionResult) -> list[SafetyEvent]:
        """Convert a detection result into zero or more safety events."""
        events: list[SafetyEvent] = []
        for det in result.detections:
            severity = _severity_for(det.label)
            if severity is Severity.INFO:
                continue
            events.append(
                SafetyEvent(
                    event_id=str(uuid.uuid4()),
                    camera_id=camera_id,
                    severity=severity,
                    message=f"Detected {det.label.value} (conf={det.confidence:.2f})",
                    detections=[det],
                    timestamp=result.timestamp,
                )
            )
        return events

    def to_alerts(self, events: list[SafetyEvent]) -> list[Alert]:
        """Escalate qualifying events into operator alerts."""
        alerts: list[Alert] = []
        for event in events:
            max_conf = max((d.confidence for d in event.detections), default=0.0)
            should_alert = (
                event.severity is Severity.CRITICAL
                or max_conf >= settings.ALERT_CONFIDENCE_THRESHOLD
            )
            if should_alert:
                alerts.append(Alert(**event.model_dump()))
        return alerts


monitor = SafetyMonitor()
