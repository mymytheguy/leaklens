from __future__ import annotations

import math
import re
from collections.abc import Iterable

from leaklens.models import BBox, OcrItem


def clamp_int(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def polygon_to_bbox(points: Iterable[Iterable[float]] | None, frame_w: int, frame_h: int) -> BBox:
    xs: list[float] = []
    ys: list[float] = []
    if points is None:
        return BBox(0, 0, 1, 1)
    for point in points:
        coords = list(point)
        if len(coords) >= 2:
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
    if not xs or not ys:
        return BBox(0, 0, 1, 1)
    x1 = clamp_int(min(xs), 0, max(0, frame_w - 1))
    y1 = clamp_int(min(ys), 0, max(0, frame_h - 1))
    x2 = clamp_int(max(xs), x1 + 1, frame_w)
    y2 = clamp_int(max(ys), y1 + 1, frame_h)
    return BBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1))


def union_boxes(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    out = boxes[0]
    for box in boxes[1:]:
        out = out.union(box)
    return out


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def build_char_map(items: list[OcrItem]) -> tuple[str, list[tuple[int, int, OcrItem]]]:
    """Join OCR items with spaces and map character spans back to source boxes."""
    parts: list[str] = []
    spans: list[tuple[int, int, OcrItem]] = []
    cursor = 0
    for item in items:
        text = item.text or ""
        if not text:
            continue
        if parts:
            cursor += 1
            parts.append(" ")
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, item))
    return "".join(parts), spans


def boxes_for_span(
    spans: list[tuple[int, int, OcrItem]],
    start: int,
    end: int,
    *,
    slice_glyphs: bool = True,
) -> list[BBox]:
    boxes: list[BBox] = []
    for span_start, span_end, item in spans:
        if span_end <= start or span_start >= end:
            continue
        if not slice_glyphs:
            boxes.append(item.bbox)
            continue
        local_start = max(0, start - span_start)
        local_end = min(len(item.text), end - span_start)
        boxes.append(slice_bbox(item.bbox, len(item.text), local_start, local_end))
    return boxes


def slice_bbox(bbox: BBox, text_len: int, start: int, end: int) -> BBox:
    """Approximate glyph box for a substring assuming monospace-ish terminal text."""
    if text_len <= 0:
        return bbox
    start = max(0, min(text_len, start))
    end = max(start + 1, min(text_len, end))
    x1 = bbox.x + int(round(bbox.w * start / text_len))
    x2 = bbox.x + int(round(bbox.w * end / text_len))
    return BBox(x=x1, y=bbox.y, w=max(1, x2 - x1), h=bbox.h)


def normalize_secret(text: str | None) -> str:
    return re.sub(r"[^a-z0-9@.:_+\-/]+", "", (text or "").lower())


def box_contains(outer: BBox, inner: BBox) -> bool:
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x + inner.w <= outer.x + outer.w
        and inner.y + inner.h <= outer.y + outer.h
    )


def collapse_overlapping_ocr(items: list[OcrItem]) -> list[OcrItem]:
    """Drop duplicate word/line/inverted OCR so row-joining cannot invent secrets."""
    ordered = sorted(
        items,
        key=lambda i: (-len((i.text or "").strip()), -i.confidence, -(i.bbox.w * i.bbox.h)),
    )
    kept: list[OcrItem] = []
    for item in ordered:
        text = (item.text or "").strip()
        redundant = False
        for other in kept:
            overlaps = item.bbox.iou(other.bbox) >= 0.15 or box_contains(other.bbox, item.bbox)
            if not overlaps:
                continue
            other_text = (other.text or "").strip()
            if not text or text in other_text or other_text in text:
                redundant = True
                break
        if not redundant:
            kept.append(item)
    kept.sort(key=lambda i: (i.bbox.y, i.bbox.x, -(i.bbox.w * i.bbox.h)))
    return kept


def looks_like_version(text: str) -> bool:
    return bool(re.fullmatch(r"v?\d+\.\d+\.\d+(?:\.\d+)?", text.strip(), flags=re.I))
