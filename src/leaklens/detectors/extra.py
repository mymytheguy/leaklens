from __future__ import annotations

import re

from leaklens.classify import make_detection
from leaklens.config import AppConfig
from leaklens.detectors.base import FrameContext
from leaklens.models import BBox, Detection
from leaklens.util import slice_bbox


class ExtraTermsDetector:
    """Operator-supplied terms and forced boxes. Used when a shoot has known leaks."""

    name = "extra"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.terms = [str(term).strip() for term in (config.extra_terms or []) if str(term).strip()]
        self.boxes = list(config.force_boxes or [])

    def detect(self, ctx: FrameContext) -> list[Detection]:
        hits: list[Detection] = []
        hits.extend(self._forced_boxes(ctx))
        hits.extend(self._terms(ctx))
        return hits

    def _forced_boxes(self, ctx: FrameContext) -> list[Detection]:
        hits: list[Detection] = []
        for spec in self.boxes:
            if not isinstance(spec, dict):
                continue
            t0 = float(spec.get("t0", 0.0) or 0.0)
            t1 = spec.get("t1")
            if ctx.timestamp < t0:
                continue
            if t1 is not None and ctx.timestamp > float(t1):
                continue
            try:
                box = BBox(
                    int(spec["x"]),
                    int(spec["y"]),
                    int(spec["w"]),
                    int(spec["h"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            category = str(spec.get("category") or "other_sensitive")
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=box.padded(int(spec.get("pad", 4)), ctx.width, ctx.height),
                    category=category,
                    detector="extra.force_box",
                    confidence=0.99,
                    text=str(spec.get("label") or "forced"),
                    classification="SENSITIVE",
                )
            )
        return hits

    def _terms(self, ctx: FrameContext) -> list[Detection]:
        if not self.terms:
            return []
        hits: list[Detection] = []
        pad = max(10, self.config.bbox_padding_px + 6)
        for item in ctx.ocr_items:
            haystack = item.text or ""
            if not haystack:
                continue
            lowered = haystack.lower()
            for term in self.terms:
                needle = term.lower()
                start = 0
                while True:
                    idx = lowered.find(needle, start)
                    if idx < 0:
                        break
                    end = idx + len(needle)
                    box = slice_bbox(item.bbox, max(1, len(haystack)), idx, end)
                    hits.append(
                        make_detection(
                            frame_number=ctx.frame_number,
                            timestamp=ctx.timestamp,
                            bbox=box.padded(pad, ctx.width, ctx.height),
                            category=_category_for_term(term),
                            detector="extra.term",
                            confidence=0.97,
                            text=haystack[idx:end],
                            classification="SENSITIVE",
                        )
                    )
                    start = end
        return hits


def _category_for_term(term: str) -> str:
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", term):
        return "ipv4"
    if "." in term and re.search(r"[A-Za-z]{2,24}$", term.split(".")[-1]):
        return "private_url"
    if re.search(r"[\\/]Users[\\/]", term, re.I):
        return "windows_path"
    return "username_handle"
