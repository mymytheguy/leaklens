"""Cheap every-frame visual change analysis.

This stage inspects every frame. It never skips a frame; it only decides
whether expensive OCR can reuse a previous result.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from leaklens.models import BBox


@dataclass
class ChangeResult:
    identical: bool
    full_frame: bool
    regions: list[BBox]
    mean_diff: float
    changed_area_ratio: float


def downscale_gray(frame_bgr: np.ndarray, scale: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if scale <= 1:
        return gray
    height, width = gray.shape
    return cv2.resize(
        gray,
        (max(32, width // scale), max(18, height // scale)),
        interpolation=cv2.INTER_AREA,
    )


def analyze_change(
    previous: np.ndarray | None,
    current: np.ndarray,
    *,
    frame_w: int,
    frame_h: int,
    pixel_threshold: int = 10,
    tile_cols: int = 16,
    tile_rows: int = 9,
    tile_fraction: float = 0.002,
    full_frame_ratio: float = 0.35,
    pad_px: int = 40,
    identical_mean: float = 1.0,
) -> ChangeResult:
    """Compare downscaled grayscale frames. Uncertain → treat as changed."""
    if previous is None or previous.shape != current.shape:
        return ChangeResult(
            identical=False,
            full_frame=True,
            regions=[BBox(0, 0, frame_w, frame_h)],
            mean_diff=255.0,
            changed_area_ratio=1.0,
        )

    delta = cv2.absdiff(previous, current)
    mean_diff = float(delta.mean())
    if mean_diff == 0.0:
        return ChangeResult(True, False, [], 0.0, 0.0)
    small_h, small_w = current.shape
    scale_x = frame_w / small_w
    scale_y = frame_h / small_h

    # Drop encoder speckle; keep blobs large enough to be glyphs.
    hot = cv2.morphologyEx(
        (delta >= pixel_threshold).astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    tile_w = max(1, small_w // tile_cols)
    tile_h = max(1, small_h // tile_rows)
    changed_tiles: list[BBox] = []
    changed_area = 0

    for row in range(tile_rows):
        y1 = row * tile_h
        y2 = small_h if row == tile_rows - 1 else min(small_h, y1 + tile_h)
        for col in range(tile_cols):
            x1 = col * tile_w
            x2 = small_w if col == tile_cols - 1 else min(small_w, x1 + tile_w)
            tile = hot[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            frac = float(tile.mean())
            if frac >= tile_fraction:
                full = BBox(
                    x=int(x1 * scale_x),
                    y=int(y1 * scale_y),
                    w=max(1, int((x2 - x1) * scale_x)),
                    h=max(1, int((y2 - y1) * scale_y)),
                ).padded(pad_px, frame_w, frame_h)
                changed_tiles.append(full)
                changed_area += (x2 - x1) * (y2 - y1)

    changed_area_ratio = changed_area / float(small_w * small_h)
    if not changed_tiles:
        # Speckle-only frames are unchanged. A real glyph leaves a blob.
        return ChangeResult(True, False, [], mean_diff, 0.0)

    merged = _merge_regions(changed_tiles)
    full_frame = changed_area_ratio >= full_frame_ratio or len(merged) >= max(6, tile_cols * tile_rows // 3)
    if full_frame:
        return ChangeResult(False, True, [BBox(0, 0, frame_w, frame_h)], mean_diff, changed_area_ratio)
    return ChangeResult(False, False, merged, mean_diff, changed_area_ratio)


def item_overlaps_regions(box: BBox, regions: list[BBox]) -> bool:
    for region in regions:
        if box.iou(region) > 0.02 or _intersects(box, region):
            return True
    return False


def _intersects(a: BBox, b: BBox) -> bool:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _merge_regions(boxes: list[BBox]) -> list[BBox]:
    remaining = list(boxes)
    merged: list[BBox] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[BBox] = []
            for other in remaining:
                if current.iou(other) > 0.0 or _intersects(current, other):
                    current = current.union(other)
                    changed = True
                else:
                    keep.append(other)
            remaining = keep
        merged.append(current)
    return merged
