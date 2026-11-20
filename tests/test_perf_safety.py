"""Regression tests: one-frame leaks stay visible; static frames skip OCR."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from leaklens.change import analyze_change, downscale_gray
from leaklens.config import AppConfig
from leaklens.models import BBox, OcrItem
from leaklens.pipeline import run_scan

SECRET_TEXTS = {
    0: "apt update && true",
    1: "10.9.8.7",
    2: "37.774900, -122.419400",
    3: "ghp_abcdefghijklmnopqrstuvwxyz12",
    4: "192.168.10.12",
}


def _blank(w: int = 640, h: int = 180) -> np.ndarray:
    return np.full((h, w, 3), 20, dtype=np.uint8)


def _put(frame: np.ndarray, text: str, x: int = 40, y: int = 90, scale: float = 0.7) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 90), 1, cv2.LINE_AA)


def _frame(kind: int, y: int = 90) -> np.ndarray:
    frame = _blank()
    _put(frame, SECRET_TEXTS[kind], y=y)
    frame[0, 0] = (kind, 0, 0)
    return frame


def _write(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


class CountingOcr:
    compute_label = "CPU"

    def __init__(self) -> None:
        self.full = 0
        self.region = 0
        self.templates = {}
        for kind, text in SECRET_TEXTS.items():
            gray = cv2.cvtColor(_frame(kind), cv2.COLOR_BGR2GRAY)
            self.templates[text] = gray[50:120, 20:420]

    def _identify(self, image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        best_val = -1.0
        best_text = SECRET_TEXTS[0]
        for text, tmpl in self.templates.items():
            if gray.shape[0] < tmpl.shape[0] or gray.shape[1] < tmpl.shape[1]:
                scaled = cv2.resize(tmpl, (min(tmpl.shape[1], gray.shape[1]), min(tmpl.shape[0], gray.shape[0])))
                if gray.shape[0] < scaled.shape[0] or gray.shape[1] < scaled.shape[1]:
                    continue
                tmpl = scaled
            result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            value = float(result.max())
            if value > best_val:
                best_val = value
                best_text = text
        return best_text if best_val >= 0.35 else SECRET_TEXTS[0]

    def run(self, frame_bgr):
        self.full += 1
        kind = int(frame_bgr[0, 0, 0])
        text = SECRET_TEXTS.get(kind) or self._identify(frame_bgr)
        return [OcrItem(text, 0.99, BBox(40, 70, 220, 22))], []

    def run_region(self, crop_bgr, origin, cache=None):
        self.region += 1
        text = self._identify(crop_bgr)
        return [OcrItem(text, 0.99, BBox(origin.x + 4, origin.y + 4, 180, 18))], []


def _scan(frames, ocr, name: str, **config_overrides):
    tmp = Path(tempfile.mkdtemp(prefix="leaklens_perf_"))
    video = tmp / f"{name}.mp4"
    _write(video, frames)
    out = tmp / "out"

    def iterator():
        for index, frame in enumerate(frames):
            yield index, index / 30.0, frame

    cfg = dict(save_ocr_dump=False, save_crops=False, prefer_gpu=False)
    cfg.update(config_overrides)
    return run_scan(
        video,
        output_video=out / "redacted.mp4",
        report_dir=out,
        config=AppConfig(**cfg),
        ocr_engine=ocr,
        force_cpu=True,
        frame_iter=iterator(),
        skip_export=True,
    )


class ChangeDetectionTests(unittest.TestCase):
    def test_identical_frames_are_unchanged(self) -> None:
        frame = _frame(0)
        small = downscale_gray(frame, 2)
        result = analyze_change(small, small.copy(), frame_w=640, frame_h=180)
        self.assertTrue(result.identical)

    def test_compression_noise_is_not_a_content_change(self) -> None:
        base = _frame(0)
        noisy = base.copy()
        rng = np.random.default_rng(0)
        noise = rng.integers(-3, 4, size=noisy.shape, dtype=np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = analyze_change(
            downscale_gray(base, 2),
            downscale_gray(noisy, 2),
            frame_w=640,
            frame_h=180,
        )
        self.assertTrue(result.identical)

    def test_one_frame_ip_pixels_are_changed(self) -> None:
        result = analyze_change(
            downscale_gray(_frame(0), 2),
            downscale_gray(_frame(1), 2),
            frame_w=640,
            frame_h=180,
        )
        self.assertFalse(result.identical)
        self.assertTrue(result.regions)


class OneFrameLeakTests(unittest.TestCase):
    def test_a_one_frame_ipv4(self) -> None:
        frames = [_frame(1 if i == 5 else 0) for i in range(12)]
        result = _scan(frames, CountingOcr(), "one_ip", change_full_frame_ratio=0.0)
        ip_frames = {d.frame_number for d in result.detections if "ipv4" in d.category}
        self.assertIn(5, ip_frames)

    def test_b_one_frame_gps(self) -> None:
        frames = [_frame(2 if i == 4 else 0) for i in range(10)]
        result = _scan(frames, CountingOcr(), "one_gps", change_full_frame_ratio=0.0)
        self.assertTrue(any(d.category == "gps_coordinates" for d in result.detections))
        self.assertIn(4, {d.frame_number for d in result.detections if d.category == "gps_coordinates"})

    def test_c_one_frame_token_between_similar_frames(self) -> None:
        frames = [_frame(3 if i == 6 else 0) for i in range(10)]
        result = _scan(frames, CountingOcr(), "one_token", change_full_frame_ratio=0.0)
        self.assertTrue(any(d.category == "access_token" for d in result.detections))
        self.assertIn(6, {d.frame_number for d in result.detections if d.category == "access_token"})

    def test_d_static_content_does_not_full_ocr_every_frame(self) -> None:
        frames = [_frame(1) for _ in range(80)]
        ocr = CountingOcr()
        result = _scan(frames, ocr, "static")
        stats = result.performance
        self.assertEqual(stats.frames_inspected, 80)
        self.assertLessEqual(ocr.full, 2)
        self.assertEqual(ocr.region, 0)
        self.assertGreaterEqual(stats.cache_hits, 70)
        self.assertTrue(any("ipv4" in d.category for d in result.detections))

    def test_e_scrolling_keeps_redaction(self) -> None:
        frames = [_frame(4, y=40 + i * 8) for i in range(12)]
        ocr = CountingOcr()
        result = _scan(frames, ocr, "scroll")
        ip_frames = {d.frame_number for d in result.detections if "ipv4" in d.category}
        self.assertGreaterEqual(len(ip_frames), 10)
        self.assertGreaterEqual(ocr.full + ocr.region, 1)


if __name__ == "__main__":
    unittest.main()
