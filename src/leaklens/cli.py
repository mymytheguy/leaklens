"""LeakLens command-line interface."""

from __future__ import annotations

import argparse
import shutil
import sys
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from leaklens import __version__
from leaklens.categories import category_label
from leaklens.compute import detect_compute
from leaklens.config import AppConfig
from leaklens.exceptions import LeakLensError, UnsupportedMediaError, UsageError
from leaklens.hooks import PipelineHooks
from leaklens.metrics import PerformanceStats
from leaklens.models import Detection
from leaklens.render import partial_output_path
from leaklens.scanner import run_scan
from leaklens.video_io import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    VideoInfo,
    is_image_path,
    is_video_path,
    probe_video,
)

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_UNSUPPORTED = 3
EXIT_PIPELINE = 4
EXIT_EXPORT = 5

MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES

BANNER = r"""
██╗     ███████╗ █████╗ ██╗  ██╗██╗     ███████╗███╗   ██╗███████╗
██║     ██╔════╝██╔══██╗██║ ██╔╝██║     ██╔════╝████╗  ██║██╔════╝
██║     █████╗  ███████║█████╔╝ ██║     █████╗  ██╔██╗ ██║███████╗
██║     ██╔══╝  ██╔══██║██╔═██╗ ██║     ██╔══╝  ██║╚██╗██║╚════██║
███████╗███████╗██║  ██║██║  ██╗███████╗███████╗██║ ╚████║███████╗
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
""".strip("\n")


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = max(0.0, seconds - minutes * 60)
    return f"{minutes:02d}:{remainder:06.3f}"


def normalize_input_path(raw: str) -> Path:
    cleaned = (raw or "").strip().strip('"').strip("'")
    return Path(cleaned).expanduser()


def _default_redacted_name(source: Path) -> str:
    suffix = source.suffix.lower() if is_image_path(source) else ".mp4"
    return f"{source.stem}.redacted{suffix}"


def resolve_output_paths(source: Path, output: str | None) -> tuple[Path, Path]:
    """Return (redacted media, report_dir)."""
    if not output:
        media = source.with_name(_default_redacted_name(source))
        return media, media.with_name(f"{source.stem}.leaklens")
    dest = Path(output.strip().strip('"').strip("'")).expanduser()
    if dest.exists() and dest.is_dir():
        media = dest / _default_redacted_name(source)
        return media, dest / f"{source.stem}.leaklens"
    if dest.suffix.lower() in MEDIA_SUFFIXES:
        return dest, dest.with_name(f"{dest.stem}.leaklens")
    if not dest.suffix:
        dest.mkdir(parents=True, exist_ok=True)
        media = dest / _default_redacted_name(source)
        return media, dest / f"{source.stem}.leaklens"
    return dest, dest.with_name(f"{dest.stem}.leaklens")


def parse_serve_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leaklens serve",
        description="Open a local upload page for photos and videos.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args(argv)
    args.serve = True
    args.path = None
    args.quiet = False
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "serve":
        return parse_serve_args(raw[1:])
    parser = argparse.ArgumentParser(
        prog="leaklens",
        description="Scan a photo or video locally for sensitive leaks and write a redacted copy.",
        epilog=(
            "Examples:\n"
            "  leaklens demo.mp4\n"
            "  leaklens shot.png\n"
            "  leaklens serve\n"
            "  leaklens demo.mp4 -o demo.redacted.mp4\n"
            "  leaklens demo.mp4 --strict --review\n"
            "  leaklens demo.mp4 --json findings.json\n"
            "  leaklens demo.mp4 --cpu --benchmark\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to the source photo or video. The original file is never modified.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Redacted photo/video path or directory (default: <name>.redacted.<ext>)",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Write detections JSON to FILE in addition to the report directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Maximum-recall scan (every frame, inverted + sharpened OCR)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Open review.html when the scan finishes",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress banner, progress, and detection lines",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the ASCII banner",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write detector debug logs to the terminal",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if a GPU is available",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print per-stage timings after the scan",
    )
    parser.add_argument(
        "--config",
        help="Optional YAML file overriding detector thresholds",
    )
    parser.add_argument(
        "--ocr-dump",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stride",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    return parser.parse_args(argv)


def apply_strict(config: AppConfig) -> None:
    config.frame_stride = 1
    config.ocr_run_inverted = True
    config.ocr_run_sharpen = True
    config.ocr_min_confidence = min(config.ocr_min_confidence, 0.12)
    config.track_hold_unreadable_frames = max(config.track_hold_unreadable_frames, 10)


def _ok(console: Console, label: str, value: str) -> None:
    console.print(f"[bold green][+][/] {label:<13}{value}")


def _info(console: Console, message: str) -> None:
    console.print(f"[bold cyan][*][/] {message}")


def _warn_det(console: Console, timestamp: float, category: str) -> None:
    console.print(
        f"[bold yellow][!][/] {format_timestamp(timestamp)}  {category_label(category)} detected"
    )


def _err(console: Console, message: str) -> None:
    console.print(f"[bold red][-][/] {message}")


def run_cli(args: argparse.Namespace, console: Console) -> int:
    if args.version:
        console.print(f"leaklens {__version__}")
        return EXIT_SUCCESS
    if getattr(args, "serve", False):
        from leaklens.web import serve_app

        return serve_app(host=args.host, port=args.port, open_browser=not args.quiet)

    if not args.path:
        raise UsageError("missing media path. Try: leaklens <photo_or_video>  or  leaklens serve")

    source = normalize_input_path(args.path)
    if not source.exists():
        raise UsageError(f"input file not found: {source}")
    if not source.is_file():
        raise UsageError(f"input is not a file: {source}")
    if not (is_image_path(source) or is_video_path(source)):
        raise UnsupportedMediaError(f"Unsupported media type: {source.suffix or source.name}")

    config = AppConfig.load(args.config)
    if args.stride:
        config.frame_stride = max(1, args.stride)
    if args.strict:
        apply_strict(config)
    if args.cpu:
        config.prefer_gpu = False

    output_video, report_dir = resolve_output_paths(source, args.output)
    quiet = bool(args.quiet)
    show_banner = not quiet and not args.no_banner

    try:
        info = probe_video(source)
    except Exception as exc:
        raise UnsupportedMediaError(f"Could not open media: {source}") from exc
    if info.width < 2 or info.height < 2:
        raise UnsupportedMediaError(f"Unsupported or empty media: {source}")

    if show_banner:
        console.print(f"[bold cyan]{BANNER}[/]")
        console.print(f"[dim]LeakLens v{__version__}  ·  local leak detection[/]\n")

    if not quiet:
        _ok(console, "Input:", source.name)
        _ok(console, "Resolution:", f"{info.width}x{info.height}")
        kind = "image" if is_image_path(source) else "video"
        _ok(console, "Kind:", kind)
        _ok(console, "Frames:", str(info.frame_count))
        _ok(console, "OCR:", "enabled" if args.ocr_dump is None else "replay")
        _ok(console, "Tracking:", "enabled")
        compute_label, _ = detect_compute(force_cpu=bool(args.cpu))
        _ok(console, "Compute:", compute_label)
        _ok(console, "Scan mode:", "strict" if args.strict else "standard")
        console.print()
        _info(console, "Scanning frames...")

    progress = Progress(
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
        transient=False,
    )
    scan_task = None
    render_task = None

    def on_info(info: VideoInfo) -> None:
        nonlocal scan_task, render_task
        if scan_task is None:
            total = max(1, info.frame_count)
            scan_task = progress.add_task("Scanning frames", total=total)
            render_task = progress.add_task("Rendering", total=total, visible=False)

    def on_frame(current: int, total: int, timestamp: float) -> None:
        if scan_task is not None:
            progress.update(scan_task, completed=current, total=max(1, total))

    def on_detection(det: Detection) -> None:
        return

    def on_phase(name: str, current: int | None, total: int | None) -> None:
        if name == "render" and render_task is not None:
            progress.update(render_task, visible=True)
            if current is not None:
                progress.update(render_task, completed=current, total=max(1, total or current))

    hooks = PipelineHooks(
        on_info=on_info,
        on_frame=on_frame,
        on_detection=on_detection,
        on_phase=on_phase,
    )

    created_report = not report_dir.exists()
    try:
        with progress:
            result = run_scan(
                source,
                output_video=output_video,
                report_dir=report_dir,
                config=config,
                ocr_dump_path=args.ocr_dump,
                json_path=args.json,
                hooks=hooks,
                print_summary=False,
                log_to_console=bool(args.verbose),
                force_cpu=bool(args.cpu),
            )
    except KeyboardInterrupt:
        _cleanup_partial(output_video)
        if created_report:
            shutil.rmtree(report_dir, ignore_errors=True)
        if not quiet:
            _err(console, "Interrupted. Partial output removed.")
        return EXIT_ERROR

    tracks = result.summary.get("track_count", 0)
    if not quiet:
        console.print()
        _print_detection_ranges(console, result.detections)
        _ok(console, "Scan complete", "")
        _ok(console, "Redactions:", str(tracks))
        _ok(console, "Output:", str(result.outputs["redacted_preview"]))
        _ok(console, "Report:", str(result.outputs["review"]))
        if "detections" in result.outputs:
            _ok(console, "JSON:", str(result.outputs["detections"]))
        if result.performance:
            _print_performance(console, result.performance, benchmark=bool(args.benchmark))

    if args.review:
        webbrowser.open(result.outputs["review"].resolve().as_uri())
    return EXIT_SUCCESS


def _print_detection_ranges(console: Console, detections: list[Detection]) -> None:
    grouped: dict[str, list[Detection]] = {}
    for det in detections:
        grouped.setdefault(det.category, []).append(det)
    if not grouped:
        return
    console.print()
    for category, group in sorted(grouped.items(), key=lambda item: min(d.timestamp for d in item[1])):
        first = min(d.timestamp for d in group)
        last = max(d.timestamp for d in group)
        label = category_label(category)
        console.print(f"[bold yellow][!][/] {label} detected")
        console.print(f"    {format_timestamp(first)} \u2192 {format_timestamp(last)}")


def _print_performance(console: Console, stats: PerformanceStats, *, benchmark: bool) -> None:
    console.print()
    console.print("[bold]Performance[/]")
    console.print("----------------------------")
    console.print(f"Frames inspected:    {stats.frames_inspected:>8}")
    console.print(f"Full OCR passes:     {stats.full_ocr_passes:>8}")
    console.print(f"Region OCR passes:   {stats.region_ocr_passes:>8}")
    console.print(f"Cache hits:          {stats.cache_hits:>8}")
    console.print(f"Tracked detections:  {stats.tracked_detections:>8}")
    console.print(f"Average scan speed:  {stats.average_fps:>7.1f} FPS")
    console.print(f"Processing time:     {_mmss(stats.elapsed_s):>8}")
    console.print(f"Video duration:      {_mmss(stats.video_duration_s):>8}")
    if benchmark:
        console.print()
        console.print("[bold]Stage timings[/]")
        console.print(f"Decode:          {stats.decode_s:6.1f}s")
        console.print(f"Change detect:   {stats.change_s:6.1f}s")
        console.print(f"OCR:             {stats.ocr_s:6.1f}s")
        console.print(f"Classification:  {stats.classify_s:6.1f}s")
        console.print(f"Tracking:        {stats.tracking_s:6.1f}s")
        console.print(f"Rendering:       {stats.render_s:6.1f}s")


def _mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _cleanup_partial(output_video: Path) -> None:
    partial = partial_output_path(output_video)
    if partial.exists():
        partial.unlink(missing_ok=True)
    legacy = output_video.with_name(output_video.name + ".partial")
    if legacy.exists():
        legacy.unlink(missing_ok=True)
    if output_video.exists() and output_video.stat().st_size == 0:
        output_video.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    console = Console(stderr=False, highlight=False)
    try:
        args = parse_args(argv)
        return run_cli(args, console)
    except LeakLensError as exc:
        console.print(f"[bold red][-][/] {exc}", highlight=False)
        return int(getattr(exc, "exit_code", EXIT_ERROR))
    except KeyboardInterrupt:
        console.print("[bold red][-][/] Interrupted.", highlight=False)
        return EXIT_ERROR
    except Exception as exc:  # pragma: no cover - last-resort guard
        console.print(f"[bold red][-][/] {exc}", highlight=False)
        return EXIT_ERROR


def legacy_main(argv: list[str] | None = None) -> int:
    """Directory-based entry point used by redact_video.py."""
    parser = argparse.ArgumentParser(
        description="Scan a video locally for potential sensitive leaks and emit a redacted preview.",
        epilog="Prefer: leaklens input.mp4",
    )
    parser.add_argument("input", help="Path to the source video. The original file is never modified.")
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Directory for redacted_preview.mp4, detections.json, and review.html (default: output)",
    )
    parser.add_argument("--config", help="Optional YAML file overriding detector thresholds")
    parser.add_argument("--stride", type=int, help="Process every Nth frame (default 1 = every frame)")
    parser.add_argument("--no-invert", action="store_true", help="Skip the inverted OCR pass")
    parser.add_argument("--ocr-dump", help="Replay detectors from an existing ocr_dump.jsonl")
    args = parser.parse_args(argv)
    config = AppConfig.load(args.config)
    if args.stride:
        config.frame_stride = max(1, args.stride)
    if args.no_invert:
        config.ocr_run_inverted = False
    from leaklens.pipeline import run_pipeline

    run_pipeline(args.input, args.output, config, ocr_dump_path=args.ocr_dump, print_summary=True)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
