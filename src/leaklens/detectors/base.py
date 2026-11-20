from __future__ import annotations

from typing import Protocol

from leaklens.config import AppConfig
from leaklens.models import Detection, OcrItem


class FrameContext:
    def __init__(
        self,
        *,
        frame_bgr,
        frame_number: int,
        timestamp: float,
        width: int,
        height: int,
        ocr_items: list[OcrItem],
        ocr_errors: list[str],
        config: AppConfig,
    ) -> None:
        self.frame_bgr = frame_bgr
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.width = width
        self.height = height
        self.ocr_items = ocr_items
        self.ocr_errors = ocr_errors
        self.config = config

    @property
    def full_text(self) -> str:
        return " ".join(item.text for item in self.ocr_items if item.text)


class Detector(Protocol):
    name: str

    def detect(self, ctx: FrameContext) -> list[Detection]:
        ...
