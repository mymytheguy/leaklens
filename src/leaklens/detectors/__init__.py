from __future__ import annotations

from leaklens.config import AppConfig
from leaklens.detectors.base import Detector
from leaklens.detectors.extra import ExtraTermsDetector
from leaklens.detectors.heuristic import HeuristicDetector
from leaklens.detectors.regex import RegexDetector
from leaklens.detectors.visual import VisualDetector


def build_detectors(config: AppConfig) -> list[Detector]:
    """Ordered detector stack. Additional detectors can be appended here."""
    return [
        RegexDetector(config),
        HeuristicDetector(config),
        ExtraTermsDetector(config),
        VisualDetector(config),
    ]
