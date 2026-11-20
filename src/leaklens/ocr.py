from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR

from leaklens.compute import detect_compute
from leaklens.config import AppConfig
from leaklens.models import BBox, OcrItem
from leaklens.ocr_cache import OcrRegionCache
from leaklens.util import collapse_overlapping_ocr, polygon_to_bbox

logger = logging.getLogger(__name__)


class LocalOcrEngine:
    """Local RapidOCR wrapper. Model is loaded once and kept for the whole scan."""

    def __init__(self, config: AppConfig, *, force_cpu: bool = False) -> None:
        self.config = config
        self.compute_label, gpu_params = detect_compute(force_cpu=force_cpu or not getattr(config, "prefer_gpu", True))
        cfg_path = Path(__file__).resolve().parent / "assets" / "ocr_config.yaml"
        params = {
            "Global.text_score": float(config.ocr_min_confidence),
            "Global.log_level": "error",
            "Global.max_side_len": 2560,
        }
        params.update(gpu_params)
        try:
            self.engine = RapidOCR(
                config_path=str(cfg_path) if cfg_path.exists() else None,
                params=params,
            )
        except Exception:
            logger.exception("Failed to init RapidOCR with GPU/custom config; using CPU defaults")
            self.compute_label = "CPU"
            self.engine = RapidOCR(params={"Global.log_level": "error"})

    def run(self, frame_bgr: np.ndarray) -> tuple[list[OcrItem], list[str]]:
        height, width = frame_bgr.shape[:2]
        items, errors = self._run_image(frame_bgr, width, height, scale=1.0)
        return collapse_overlapping_ocr(_dedupe_ocr(items)), errors

    def run_region(
        self,
        crop_bgr: np.ndarray,
        origin: BBox,
        cache: OcrRegionCache | None = None,
    ) -> tuple[list[OcrItem], list[str]]:
        if crop_bgr.size == 0:
            return [], []
        if cache is not None:
            cached = cache.get(crop_bgr)
            if cached is not None:
                return _offset_items(cached, origin), []

        height, width = crop_bgr.shape[:2]
        image = crop_bgr
        scale = 1.0
        if height < 48:
            scale = 2.0
            image = cv2.resize(crop_bgr, (max(8, int(width * scale)), max(8, int(height * scale))))
        items, errors = self._run_image(image, image.shape[1], image.shape[0], scale=scale)
        local = []
        for item in items:
            bbox = item.bbox
            if scale != 1.0:
                bbox = BBox(
                    x=int(bbox.x / scale),
                    y=int(bbox.y / scale),
                    w=max(1, int(bbox.w / scale)),
                    h=max(1, int(bbox.h / scale)),
                )
            local.append(OcrItem(item.text, item.confidence, bbox, item.source, item.inverted))
        if cache is not None:
            cache.put(crop_bgr, local)
        return _offset_items(local, origin), errors

    def _run_image(
        self,
        image_bgr: np.ndarray,
        width: int,
        height: int,
        *,
        scale: float,
    ) -> tuple[list[OcrItem], list[str]]:
        items: list[OcrItem] = []
        errors: list[str] = []
        variants = [("normal", image_bgr)]
        if self.config.ocr_run_inverted:
            variants.append(("inverted", cv2.bitwise_not(image_bgr)))
        if getattr(self.config, "ocr_run_sharpen", False):
            variants.append(("sharpen", _sharpen(image_bgr)))

        for label, image in variants:
            try:
                result = self.engine(
                    image,
                    return_word_box=True,
                    text_score=self.config.ocr_min_confidence,
                )
            except Exception as exc:
                logger.error("OCR %s pass failed: %s", label, exc)
                errors.append(f"ocr_{label}: {exc}")
                continue
            items.extend(_parse_output(result, width, height, inverted=label != "normal"))
        return items, errors


def _offset_items(items: list[OcrItem], origin: BBox) -> list[OcrItem]:
    out: list[OcrItem] = []
    for item in items:
        box = BBox(item.bbox.x + origin.x, item.bbox.y + origin.y, item.bbox.w, item.bbox.h)
        out.append(OcrItem(item.text, item.confidence, box, item.source, item.inverted))
    return out


def _parse_output(result, width: int, height: int, inverted: bool) -> list[OcrItem]:
    items: list[OcrItem] = []
    if result is None:
        return items

    try:
        for word in _iter_words(getattr(result, "word_results", None)):
            text, score, box = _unpack_word(word)
            if not text or len(text.strip()) <= 1 or not box:
                continue
            items.append(
                OcrItem(
                    text=text,
                    confidence=score,
                    bbox=polygon_to_bbox(box, width, height),
                    source="rapidocr_word",
                    inverted=inverted,
                )
            )
    except Exception as exc:
        logger.debug("word_results parse skipped: %s", exc)

    txts = _as_seq(getattr(result, "txts", None))
    scores = _as_seq(getattr(result, "scores", None))
    boxes = _as_seq(getattr(result, "boxes", None))
    for idx, text in enumerate(txts):
        if not text:
            continue
        score = float(scores[idx]) if idx < len(scores) else 0.0
        box = boxes[idx] if idx < len(boxes) else [[0, 0], [1, 0], [1, 1], [0, 1]]
        items.append(
            OcrItem(
                text=str(text),
                confidence=score,
                bbox=polygon_to_bbox(box, width, height),
                source="rapidocr_line",
                inverted=inverted,
            )
        )
    return items


def _as_seq(value) -> list:
    if value is None:
        return []
    return list(value)


def _iter_words(word_results):
    """Yield RapidOCR word tuples. Layout varies by version: lines-of-words or flat words."""
    if not word_results:
        return
    for entry in word_results:
        if not entry:
            continue
        if _looks_like_word(entry):
            yield entry
            continue
        if isinstance(entry, (str, bytes, float, int, np.floating, np.integer)):
            continue
        try:
            nested = list(entry)
        except TypeError:
            continue
        for word in nested:
            if _looks_like_word(word):
                yield word


def _looks_like_word(value) -> bool:
    if isinstance(value, dict):
        return True
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) < 2:
        return False
    return isinstance(value[0], str)


def _unpack_word(word) -> tuple[str, float, list]:
    if isinstance(word, dict):
        return str(word.get("text") or ""), float(word.get("score") or 0), word.get("box") or []
    if isinstance(word, (float, int, np.floating, np.integer, str, bytes)):
        return (str(word) if isinstance(word, str) else ""), 0.0, []
    try:
        length = len(word)
    except TypeError:
        return "", 0.0, []
    text = word[0] if length > 0 else ""
    score = word[1] if length > 1 else 0.0
    box = word[2] if length > 2 else []
    return str(text or ""), float(score or 0.0), box or []


def _dedupe_ocr(items: list[OcrItem]) -> list[OcrItem]:
    seen: set[tuple] = set()
    unique: list[OcrItem] = []
    for item in items:
        key = (item.text, item.bbox.x, item.bbox.y, item.bbox.w, item.bbox.h, item.inverted)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _sharpen(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(sharp)
    return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
