from __future__ import annotations

import unittest

from leaklens.config import AppConfig
from leaklens.detectors.base import FrameContext
from leaklens.detectors.extra import ExtraTermsDetector
from leaklens.models import BBox, OcrItem


class ExtraTermsTests(unittest.TestCase):
    def test_term_and_force_box(self) -> None:
        config = AppConfig(
            extra_terms=["example-user"],
            force_boxes=[{"x": 10, "y": 20, "w": 80, "h": 16, "t0": 1.0, "t1": 2.0}],
        )
        detector = ExtraTermsDetector(config)
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=3,
            timestamp=1.5,
            width=640,
            height=480,
            ocr_items=[OcrItem(text="login example-user here", confidence=0.9, bbox=BBox(0, 40, 200, 20))],
            ocr_errors=[],
            config=config,
        )
        cats = {d.detector: d for d in detector.detect(ctx)}
        self.assertIn("extra.term", cats)
        self.assertIn("extra.force_box", cats)

    def test_ignore_and_review_gate(self) -> None:
        from leaklens.classify import detection_should_redact, make_detection

        keep = make_detection(
            frame_number=0,
            timestamp=0,
            bbox=BBox(0, 0, 10, 10),
            category="username_handle",
            detector="extra.term",
            confidence=0.9,
            text="secret-user",
            classification="SENSITIVE",
        )
        kali = make_detection(
            frame_number=0,
            timestamp=0,
            bbox=BBox(0, 0, 10, 10),
            category="full_name",
            detector="regex.full_name",
            confidence=0.5,
            text="Kali Linux",
            classification="REVIEW_REQUIRED",
        )
        config = AppConfig(ignore_texts=["kali"], redact_review_required=False)
        self.assertTrue(detection_should_redact(keep, config))
        self.assertFalse(detection_should_redact(kali, config))
