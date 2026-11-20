from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from leaklens.config import AppConfig
from leaklens.models import BBox, Detection
from leaklens.ocr import _parse_output
from leaklens.render import partial_output_path, render_redacted_image, render_redacted_video


class OcrParseTests(unittest.TestCase):
    def test_flat_word_triples_do_not_crash(self) -> None:
        box = [[0, 0], [40, 0], [40, 12], [0, 12]]
        result = SimpleNamespace(
            word_results=(("10.1.2.3", 0.9, box), ("ok", 0.8, box)),
            txts=("10.1.2.3",),
            scores=(0.9,),
            boxes=(box,),
        )
        items = _parse_output(result, 200, 80, inverted=False)
        texts = {item.text for item in items}
        self.assertIn("10.1.2.3", texts)

    def test_float_entries_are_ignored(self) -> None:
        result = SimpleNamespace(
            word_results=(("token", 0.99, [[0, 0], [8, 0], [8, 8], [0, 8]]),),
            txts=None,
            scores=None,
            boxes=None,
        )
        items = _parse_output(result, 100, 40, inverted=False)
        self.assertTrue(any(item.text == "token" for item in items))


class RenderPathTests(unittest.TestCase):
    def test_partial_keeps_mp4_suffix(self) -> None:
        dest = Path(r"C:\Videos\clip.redacted.mp4")
        partial = partial_output_path(dest)
        self.assertEqual(partial.name, "clip.redacted.partial.mp4")
        self.assertTrue(str(partial).endswith(".mp4"))
        self.assertFalse(str(partial).endswith(".mp4.partial"))

    def test_render_writes_mp4(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="leaklens_render_"))
        source = tmp / "src.mp4"
        dest = tmp / "out.redacted.mp4"
        frames = [np.full((64, 96, 3), 30, dtype=np.uint8) for _ in range(6)]
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (96, 64))
        for frame in frames:
            writer.write(frame)
        writer.release()
        dets = [
            Detection(
                frame_number=2,
                timestamp=0.1,
                bbox=BBox(10, 10, 20, 12),
                category="ipv4_private",
                detector="test",
                confidence=0.9,
                text="10.0.0.1",
            )
        ]
        render_redacted_video(source, dest, dets, AppConfig(), 15.0, frame_count=6)
        self.assertTrue(dest.exists())
        self.assertGreater(dest.stat().st_size, 32)
        self.assertFalse(partial_output_path(dest).exists())

    def test_render_writes_png(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="leaklens_img_"))
        source = tmp / "src.png"
        dest = tmp / "out.redacted.png"
        frame = np.full((48, 80, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(source), frame)
        dets = [
            Detection(
                frame_number=0,
                timestamp=0.0,
                bbox=BBox(8, 8, 24, 12),
                category="ipv4_private",
                detector="test",
                confidence=0.9,
                text="10.0.0.1",
            )
        ]
        render_redacted_image(source, dest, dets, AppConfig())
        self.assertTrue(dest.exists())
        self.assertGreater(dest.stat().st_size, 16)
        out = cv2.imread(str(dest))
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[:2], (48, 80))
