from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerformanceStats:
    frames_inspected: int = 0
    full_ocr_passes: int = 0
    region_ocr_passes: int = 0
    cache_hits: int = 0
    tracked_detections: int = 0
    decode_s: float = 0.0
    change_s: float = 0.0
    ocr_s: float = 0.0
    classify_s: float = 0.0
    tracking_s: float = 0.0
    render_s: float = 0.0
    elapsed_s: float = 0.0
    video_duration_s: float = 0.0
    compute: str = "CPU"
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def average_fps(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.frames_inspected / self.elapsed_s

    def to_dict(self) -> dict:
        return {
            "frames_inspected": self.frames_inspected,
            "full_ocr_passes": self.full_ocr_passes,
            "region_ocr_passes": self.region_ocr_passes,
            "cache_hits": self.cache_hits,
            "tracked_detections": self.tracked_detections,
            "average_fps": round(self.average_fps, 2),
            "elapsed_s": round(self.elapsed_s, 3),
            "video_duration_s": round(self.video_duration_s, 3),
            "compute": self.compute,
            "stages": {
                "decode": round(self.decode_s, 3),
                "change_detect": round(self.change_s, 3),
                "ocr": round(self.ocr_s, 3),
                "classification": round(self.classify_s, 3),
                "tracking": round(self.tracking_s, 3),
                "rendering": round(self.render_s, 3),
            },
        }
