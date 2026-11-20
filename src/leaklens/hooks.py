from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from leaklens.models import Detection
from leaklens.video_io import VideoInfo


@dataclass
class PipelineHooks:
    """Optional UI callbacks. Detection behavior does not depend on these."""

    on_info: Callable[[VideoInfo], None] | None = None
    on_frame: Callable[[int, int, float], None] | None = None
    on_detection: Callable[[Detection], None] | None = None
    on_phase: Callable[[str, int | None, int | None], None] | None = None
