"""In-memory OCR cache keyed by a conservative visual fingerprint."""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from leaklens.models import OcrItem


def region_fingerprint(crop_bgr: np.ndarray) -> tuple[int, bytes]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    tiny = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    mean = float(tiny.mean())
    bits = 0
    for value in tiny.flatten():
        bits = (bits << 1) | int(value >= mean)
    return bits, tiny.tobytes()


class OcrRegionCache:
    def __init__(self, max_entries: int = 256, max_thumb_diff: float = 3.0) -> None:
        self.max_entries = max_entries
        self.max_thumb_diff = max_thumb_diff
        self._store: OrderedDict[int, tuple[bytes, list[OcrItem]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, crop_bgr: np.ndarray) -> list[OcrItem] | None:
        key, thumb = region_fingerprint(crop_bgr)
        packed = self._store.get(key)
        if packed is None:
            self.misses += 1
            return None
        prev_thumb, items = packed
        diff = _thumb_mae(prev_thumb, thumb)
        if diff > self.max_thumb_diff:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return [OcrItem(i.text, i.confidence, i.bbox, i.source, i.inverted) for i in items]

    def put(self, crop_bgr: np.ndarray, items: list[OcrItem]) -> None:
        key, thumb = region_fingerprint(crop_bgr)
        clones = [OcrItem(i.text, i.confidence, i.bbox, i.source, i.inverted) for i in items]
        self._store[key] = (thumb, clones)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


def _thumb_mae(a: bytes, b: bytes) -> float:
    left = np.frombuffer(a, dtype=np.uint8)
    right = np.frombuffer(b, dtype=np.uint8)
    if left.size != right.size or left.size == 0:
        return 255.0
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))
