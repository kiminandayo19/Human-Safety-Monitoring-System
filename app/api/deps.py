"""Shared API dependencies."""

from app.services.detector import SafetyDetector, detector
from app.services.safety_monitor import SafetyMonitor, monitor


def get_detector() -> SafetyDetector:
    """Provide the shared detector instance."""
    return detector


def get_monitor() -> SafetyMonitor:
    """Provide the shared safety monitor instance."""
    return monitor
