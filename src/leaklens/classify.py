from __future__ import annotations

from leaklens.config import AppConfig
from leaklens.models import BBox, Detection
from leaklens.util import normalize_secret


def make_detection(
    *,
    frame_number: int,
    timestamp: float,
    bbox: BBox,
    category: str,
    detector: str,
    confidence: float,
    text: str | None,
    classification: str,
    extra: dict | None = None,
) -> Detection:
    label = "REVIEW_REQUIRED" if classification == "REVIEW_REQUIRED" else "SENSITIVE"
    return Detection(
        frame_number=frame_number,
        timestamp=timestamp,
        bbox=bbox,
        category=category,
        detector=detector,
        confidence=round(float(confidence), 4),
        text=text,
        normalized_text=normalize_secret(text),
        classification=label,
        origin="direct",
        review_required=label == "REVIEW_REQUIRED",
        extra=extra or {},
    )


def detection_ignored(det: Detection, config: AppConfig) -> bool:
    haystack = f"{det.text or ''} {det.normalized_text or ''}".lower()
    for needle in config.ignore_texts or []:
        token = str(needle).strip().lower()
        if token and token in haystack:
            return True
    return False


def detection_should_redact(det: Detection, config: AppConfig) -> bool:
    if detection_ignored(det, config):
        return False
    if not config.redact_review_required and (
        det.review_required or det.classification == "REVIEW_REQUIRED"
    ):
        return False
    return True
