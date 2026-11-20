from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def padded(self, pad: int, frame_w: int, frame_h: int) -> BBox:
        x1 = max(0, self.x - pad)
        y1 = max(0, self.y - pad)
        x2 = min(frame_w, self.x + self.w + pad)
        y2 = min(frame_h, self.y + self.h + pad)
        return BBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1))

    def iou(self, other: BBox) -> float:
        ax1, ay1, ax2, ay2 = self.as_xyxy()
        bx1, by1, bx2, by2 = other.as_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union else 0.0

    def union(self, other: BBox) -> BBox:
        ax1, ay1, ax2, ay2 = self.as_xyxy()
        bx1, by1, bx2, by2 = other.as_xyxy()
        x1, y1 = min(ax1, bx1), min(ay1, by1)
        x2, y2 = max(ax2, bx2), max(ay2, by2)
        return BBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1))

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class OcrItem:
    text: str
    confidence: float
    bbox: BBox
    source: str = "rapidocr"
    inverted: bool = False


@dataclass
class Detection:
    frame_number: int
    timestamp: float
    bbox: BBox
    category: str
    detector: str
    confidence: float
    text: str | None = None
    normalized_text: str | None = None
    classification: str = "SENSITIVE"
    origin: str = "direct"
    review_required: bool = False
    temporal_extended: bool = False
    track_id: int | None = None
    first_frame_detected: int | None = None
    last_frame_redacted: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def clone_on_frame(self, frame_number: int, timestamp: float, origin: str = "temporal_gap") -> Detection:
        return replace(
            self,
            frame_number=frame_number,
            timestamp=timestamp,
            temporal_extended=True,
            origin=origin,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload
