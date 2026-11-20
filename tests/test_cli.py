from __future__ import annotations

import io
import unittest
from pathlib import Path

from rich.console import Console

from leaklens.cli import (
    EXIT_SUCCESS,
    EXIT_USAGE,
    format_timestamp,
    main,
    normalize_input_path,
    parse_args,
    resolve_output_paths,
    run_cli,
    _print_detection_ranges,
)
from leaklens.models import BBox, Detection


class CliTests(unittest.TestCase):
    def test_version(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, highlight=False, color_system=None, force_terminal=False)
        code = run_cli(parse_args(["--version"]), console)
        self.assertEqual(code, EXIT_SUCCESS)
        self.assertIn("0.1.0", buf.getvalue())

    def test_missing_path(self) -> None:
        self.assertEqual(main([]), EXIT_USAGE)

    def test_missing_file(self) -> None:
        code = main(["C:/definitely/not/a/real/video.mp4"])
        self.assertEqual(code, EXIT_USAGE)

    def test_quoted_windows_path(self) -> None:
        path = normalize_input_path(r'"C:\Users\Name\Videos\recording.mp4"')
        self.assertEqual(path, Path(r"C:\Users\Name\Videos\recording.mp4"))

    def test_default_output_name(self) -> None:
        source = Path(r"C:\Users\Name\Videos\recording.mp4")
        video, report = resolve_output_paths(source, None)
        self.assertEqual(video.name, "recording.redacted.mp4")
        self.assertEqual(report.name, "recording.leaklens")

    def test_default_image_output_name(self) -> None:
        source = Path(r"C:\Users\Name\Downloads\capture.png")
        media, report = resolve_output_paths(source, None)
        self.assertEqual(media.name, "capture.redacted.png")
        self.assertEqual(report.name, "capture.leaklens")

    def test_output_file_flag(self) -> None:
        source = Path("demo.mp4")
        video, report = resolve_output_paths(source, "out/custom.mp4")
        self.assertEqual(video.name, "custom.mp4")
        self.assertEqual(report.name, "custom.leaklens")

    def test_serve_subcommand(self) -> None:
        args = parse_args(["serve", "--port", "9001"])
        self.assertTrue(args.serve)
        self.assertEqual(args.port, 9001)
        self.assertEqual(args.host, "127.0.0.1")

    def test_help_lists_core_flags(self) -> None:
        args = parse_args(["demo.mp4", "--strict", "--review", "--quiet", "--no-banner", "--cpu", "--benchmark"])
        self.assertTrue(args.cpu)
        self.assertTrue(args.benchmark)

    def test_timestamp_format(self) -> None:
        self.assertEqual(format_timestamp(4.183), "00:04.183")
        self.assertEqual(format_timestamp(9.516), "00:09.516")

    def test_detection_ranges_merge_by_category(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, highlight=False, color_system=None, force_terminal=False)
        dets = [
            Detection(0, 0.0, BBox(1, 1, 10, 10), "autofill_suggestion", "t", 0.9, track_id=1),
            Detection(2, 0.04, BBox(1, 1, 10, 10), "autofill_suggestion", "t", 0.9, track_id=2),
            Detection(3, 0.06, BBox(1, 1, 10, 10), "email", "t", 0.9, track_id=3),
            Detection(8, 0.18, BBox(1, 1, 10, 10), "email", "t", 0.9, track_id=4),
        ]
        _print_detection_ranges(console, dets)
        text = buf.getvalue()
        self.assertEqual(text.count("Autofill suggestion detected"), 1)
        self.assertEqual(text.count("Email address detected"), 1)
        self.assertIn("00:00.000 → 00:00.040", text)
        self.assertIn("00:00.060 → 00:00.180", text)

    def test_debug_artifacts_are_off_by_default(self) -> None:
        from leaklens.config import AppConfig

        config = AppConfig()
        self.assertFalse(config.save_ocr_dump)
        self.assertFalse(config.save_crops)


if __name__ == "__main__":
    unittest.main()
