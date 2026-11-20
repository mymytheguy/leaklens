from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    """All thresholds are intentionally conservative (prefer false positives)."""

    frame_stride: int = 1
    ocr_min_confidence: float = 0.15
    ocr_run_inverted: bool = False
    ocr_run_sharpen: bool = False
    review_confidence_threshold: float = 0.75
    bbox_padding_px: int = 6
    expand_match_to_ocr_line: bool = False
    blur_kernel: int = 51
    blur_extra_pixelate: int = 12
    track_iou_threshold: float = 0.25
    track_max_gap_frames: int = 8
    track_extend_unmatched: bool = True
    track_hold_unreadable_frames: int = 10
    face_scale_factor: float = 1.05
    face_min_neighbors: int = 3
    face_min_size: int = 24
    name_min_confidence: float = 0.55
    handle_min_len: int = 3
    high_entropy_min_len: int = 20
    high_entropy_threshold: float = 3.5
    save_ocr_dump: bool = False
    save_crops: bool = False
    max_crops: int = 400
    prefer_gpu: bool = True
    change_scale: int = 2
    change_pixel_threshold: int = 10
    change_tile_fraction: float = 0.002
    change_full_frame_ratio: float = 0.35
    change_pad_px: int = 40
    ocr_cache_size: int = 256
    extra_terms: list[str] = field(default_factory=list)
    force_boxes: list[dict] = field(default_factory=list)
    ignore_texts: list[str] = field(default_factory=list)
    detect_faces: bool = True
    redact_review_required: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> AppConfig:
        cfg = cls()
        if not data:
            return cfg
        valid = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in valid:
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> AppConfig:
        if not path:
            return cls()
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
