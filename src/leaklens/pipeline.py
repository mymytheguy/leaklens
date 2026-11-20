from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from leaklens.change import ChangeResult, analyze_change, downscale_gray, item_overlaps_regions
from leaklens.classify import detection_ignored
from leaklens.config import AppConfig
from leaklens.detectors.extra import ExtraTermsDetector
from leaklens.detectors.heuristic import HeuristicDetector
from leaklens.detectors.regex import RegexDetector
from leaklens.detectors.visual import VisualDetector
from leaklens.detectors.base import FrameContext
from leaklens.exceptions import PipelineError, UnsupportedMediaError
from leaklens.hooks import PipelineHooks
from leaklens.metrics import PerformanceStats
from leaklens.models import BBox, Detection, OcrItem
from leaklens.ocr import LocalOcrEngine
from leaklens.ocr_cache import OcrRegionCache
from leaklens.render import render_redacted_media
from leaklens.report import (
    load_ocr_dump,
    ocr_items_payload,
    print_console_summary,
    representative_tracks,
    summarize,
    write_detections_json,
    write_ocr_dump,
    write_review_html,
)
from leaklens.tracking import apply_temporal_tracking
from leaklens.util import collapse_overlapping_ocr, union_boxes
from leaklens.video_io import VideoInfo, grab_frames, iter_frames, probe_video

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    source: Path
    video_info: VideoInfo
    detections: list[Detection]
    errors: list[str]
    outputs: dict[str, Path]
    summary: dict[str, Any]
    performance: PerformanceStats | None = None


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    config: AppConfig,
    ocr_dump_path: str | Path | None = None,
    *,
    hooks: PipelineHooks | None = None,
    print_summary: bool = True,
    log_to_console: bool = False,
    ocr_engine: LocalOcrEngine | None = None,
    force_cpu: bool = False,
) -> dict[str, Path]:
    """Backwards-compatible directory-based run used by tests and redact_video.py."""
    out = Path(output_dir)
    source = Path(input_path)
    from leaklens.video_io import is_image_path

    preview_name = f"redacted_preview{source.suffix.lower()}" if is_image_path(source) else "redacted_preview.mp4"
    result = run_scan(
        input_path,
        output_video=out / preview_name,
        report_dir=out,
        config=config,
        ocr_dump_path=ocr_dump_path,
        hooks=hooks,
        print_summary=print_summary,
        log_to_console=log_to_console,
        ocr_engine=ocr_engine,
        force_cpu=force_cpu,
    )
    return result.outputs


def run_scan(
    input_path: str | Path,
    *,
    output_video: str | Path,
    report_dir: str | Path,
    config: AppConfig,
    ocr_dump_path: str | Path | None = None,
    json_path: str | Path | None = None,
    hooks: PipelineHooks | None = None,
    print_summary: bool = False,
    log_to_console: bool = False,
    ocr_engine: LocalOcrEngine | None = None,
    force_cpu: bool = False,
    frame_iter=None,
    skip_export: bool = False,
) -> ScanResult:
    """Run the existing detector stack with cheap every-frame change analysis."""
    started = time.perf_counter()
    stats = PerformanceStats()
    source = Path(input_path).expanduser().resolve()
    redacted = Path(output_video).expanduser().resolve()
    out = Path(report_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    redacted.parent.mkdir(parents=True, exist_ok=True)
    crops = out / "crops"
    crops.mkdir(exist_ok=True)
    log_path = out / "run.log"
    _configure_logging(log_path, console=log_to_console)

    logger.info("Starting LeakLens scan on %s", source)
    try:
        info = probe_video(source)
    except Exception as exc:
        raise UnsupportedMediaError(f"Could not open video: {source}") from exc
    if info.width < 2 or info.height < 2:
        raise UnsupportedMediaError(f"Unsupported or empty video: {source}")
    stats.video_duration_s = info.duration_s
    logger.info("Video %sx%s fps=%s frames=%s", info.width, info.height, info.fps, info.frame_count)
    if hooks and hooks.on_info:
        hooks.on_info(info)

    cached_ocr = None
    if ocr_dump_path:
        dump = Path(ocr_dump_path).expanduser().resolve()
        logger.info("Replaying detectors from OCR dump %s (skipping live OCR)", dump)
        cached_ocr = load_ocr_dump(dump)

    try:
        ocr = ocr_engine
        if cached_ocr is None and ocr is None:
            ocr = LocalOcrEngine(config, force_cpu=force_cpu)
        if ocr is not None:
            stats.compute = ocr.compute_label
        regex = RegexDetector(config)
        heuristic = HeuristicDetector(config)
        extra = ExtraTermsDetector(config)
        visual = VisualDetector(config)
    except Exception as exc:
        raise PipelineError(f"Failed to initialize detectors: {exc}") from exc

    detections: list[Detection] = []
    errors: list[str] = []
    ocr_frames: list[dict] = []
    ocr_by_frame: dict[int, list] = {}
    ocr_item_count = 0
    failed_frames = 0
    total = max(1, info.frame_count)
    if hooks and hooks.on_phase:
        hooks.on_phase("scan", 0, total)

    prev_small = None
    prev_items: list[OcrItem] | None = None
    prev_text_dets: list[Detection] = []
    prev_visual_dets: list[Detection] = []
    region_cache = OcrRegionCache(max_entries=config.ocr_cache_size)
    saw_omnibox_url = False

    try:
        for frame_number, timestamp, frame in (
            frame_iter if frame_iter is not None else iter_frames(source, config.frame_stride)
        ):
            stats.frames_inspected += 1
            if hooks and hooks.on_frame:
                hooks.on_frame(frame_number + 1, total, timestamp)

            ocr_errors: list[str] = []
            reused = False
            if cached_ocr is not None:
                items = cached_ocr.get(frame_number, [])
                if frame_number not in cached_ocr:
                    ocr_errors = [f"ocr dump missing frame {frame_number}"]
                    errors.append(ocr_errors[0])
                text_dets, visual_dets = _classify_frame(
                    frame, frame_number, timestamp, info, config, items, ocr_errors,
                    regex, heuristic, extra, visual, run_visual=True,
                )
            else:
                change_mark = time.perf_counter()
                small = downscale_gray(frame, config.change_scale)
                change = analyze_change(
                    prev_small,
                    small,
                    frame_w=info.width,
                    frame_h=info.height,
                    pixel_threshold=config.change_pixel_threshold,
                    tile_fraction=config.change_tile_fraction,
                    full_frame_ratio=config.change_full_frame_ratio,
                    pad_px=config.change_pad_px,
                )
                stats.change_s += time.perf_counter() - change_mark
                prev_small = small

                sticky = _sticky_scan_regions(
                    prev_text_dets,
                    info.width,
                    info.height,
                    include_omnibox=saw_omnibox_url,
                )
                if change.identical and prev_items is not None and not sticky:
                    reused = True
                    stats.cache_hits += 1
                    items = prev_items
                    text_dets = _restamp(prev_text_dets, frame_number, timestamp)
                    visual_dets = _restamp(prev_visual_dets, frame_number, timestamp)
                else:
                    if change.identical and sticky:
                        change = ChangeResult(
                            identical=False,
                            full_frame=False,
                            regions=sticky,
                            mean_diff=change.mean_diff,
                            changed_area_ratio=max(change.changed_area_ratio, 0.02),
                        )
                    elif not change.full_frame:
                        change = ChangeResult(
                            identical=False,
                            full_frame=False,
                            regions=list(change.regions) + [_omnibox_region(info.width, info.height)],
                            mean_diff=change.mean_diff,
                            changed_area_ratio=change.changed_area_ratio,
                        )
                    ocr_mark = time.perf_counter()
                    try:
                        items, ocr_errors, full_pass = _ocr_for_change(
                            ocr, frame, change, prev_items, region_cache
                        )
                    except Exception as exc:
                        failed_frames += 1
                        message = f"frame {frame_number}: OCR crashed: {exc}"
                        logger.exception(message)
                        errors.append(message)
                        if prev_items is not None:
                            items, ocr_errors, full_pass = prev_items, [str(exc)], False
                            reused = True
                            stats.cache_hits += 1
                            text_dets = _restamp(prev_text_dets, frame_number, timestamp)
                            visual_dets = _restamp(prev_visual_dets, frame_number, timestamp)
                            stats.ocr_s += time.perf_counter() - ocr_mark
                            prev_small = small
                            ocr_by_frame[frame_number] = prev_items
                            ocr_item_count += len(prev_items)
                            ocr_frames.append(
                                {
                                    "frame_number": frame_number,
                                    "timestamp": timestamp,
                                    "errors": ocr_errors,
                                    "items": [],
                                    "reused": True,
                                }
                            )
                            detections.extend(
                                d for d in text_dets + visual_dets if not detection_ignored(d, config)
                            )
                            continue
                        items, ocr_errors, full_pass = [], [str(exc)], True
                    stats.ocr_s += time.perf_counter() - ocr_mark
                    if full_pass:
                        stats.full_ocr_passes += 1
                    else:
                        stats.region_ocr_passes += 1

                    run_visual = change.full_frame or change.changed_area_ratio >= 0.05
                    classify_mark = time.perf_counter()
                    text_dets, visual_dets = _classify_frame(
                        frame, frame_number, timestamp, info, config, items, ocr_errors,
                        regex, heuristic, extra, visual,
                        run_visual=run_visual,
                        reused_visual=prev_visual_dets,
                    )
                    stats.classify_s += time.perf_counter() - classify_mark
                    prev_items = items
                    prev_text_dets = text_dets
                    prev_visual_dets = visual_dets

            if ocr_errors:
                errors.extend(f"frame {frame_number}: {err}" for err in ocr_errors)
                if not items:
                    failed_frames += 1
                    logger.error("Frame %s could not be confidently processed", frame_number)

            items = collapse_overlapping_ocr(items)
            if _has_omnibox_url(items, info.width, info.height):
                saw_omnibox_url = True
            ocr_item_count += len(items)
            ocr_by_frame[frame_number] = items
            ocr_frames.append(
                {
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "errors": ocr_errors,
                    "items": [] if reused else ocr_items_payload(items),
                    "reused": reused,
                }
            )
            frame_dets = [
                d for d in text_dets + visual_dets if not detection_ignored(d, config)
            ]
            detections.extend(frame_dets)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Detection pipeline failed: {exc}") from exc

    track_mark = time.perf_counter()
    tracked = apply_temporal_tracking(
        detections,
        config,
        info.fps,
        ocr_by_frame=ocr_by_frame,
        frame_count=info.frame_count,
    )
    stats.tracking_s = time.perf_counter() - track_mark
    stats.tracked_detections = len({d.track_id for d in tracked if d.track_id is not None})
    crop_frame_ids = {
        det.frame_number
        for _, det in representative_tracks(tracked)[: config.max_crops]
    }
    review_frames = {} if skip_export else grab_frames(source, crop_frame_ids)

    detections_json = Path(json_path).expanduser().resolve() if json_path else out / "detections.json"
    detections_json.parent.mkdir(parents=True, exist_ok=True)
    review_html = out / "review.html"
    ocr_dump = out / "ocr_dump.jsonl"
    bundled_json = out / "detections.json" if json_path else detections_json

    if hooks and hooks.on_phase:
        hooks.on_phase("render", 0, total)
    render_error: Exception | None = None
    try:
        if not skip_export:
            render_mark = time.perf_counter()
            phase = hooks.on_phase if hooks else None
            on_frame: Callable[[int, int], None] | None = None
            if phase is not None:
                callback = phase

                def on_frame(i: int, n: int) -> None:
                    callback("render", i, n)
            try:
                render_redacted_media(
                    source,
                    redacted,
                    tracked,
                    config,
                    info.fps,
                    on_frame=on_frame,
                    frame_count=info.frame_count,
                )
            except Exception as exc:
                render_error = exc
                logger.exception("Redacted video encode failed")
            stats.render_s = time.perf_counter() - render_mark
            write_detections_json(
                bundled_json,
                source=source,
                video_meta=info.to_dict(),
                config=config.to_dict(),
                detections=tracked,
                errors=errors,
                ocr_stats={
                    "frames_seen": info.frame_count,
                    "frames_failed": failed_frames,
                    "ocr_items": ocr_item_count,
                    "ocr_dump": str(ocr_dump) if config.save_ocr_dump else None,
                    "performance": stats.to_dict(),
                },
            )
            if json_path and Path(json_path).expanduser().resolve() != bundled_json:
                detections_json.write_text(bundled_json.read_text(encoding="utf-8"), encoding="utf-8")
            if config.save_ocr_dump and cached_ocr is None:
                write_ocr_dump(ocr_dump, ocr_frames)
            write_review_html(
                review_html,
                source=source,
                detections=tracked,
                crops_dir=crops,
                frames_by_number=review_frames,
                max_crops=config.max_crops,
            )
    except Exception as exc:
        from leaklens.exceptions import ExportError

        raise ExportError(f"Failed to write reports: {exc}") from exc
    if render_error is not None:
        from leaklens.exceptions import ExportError

        raise ExportError(
            f"Reports were written to {out}, but the redacted video failed: {render_error}"
        ) from render_error

    outputs = {
        "redacted_preview": redacted,
        "detections": detections_json if json_path else bundled_json,
        "review": review_html,
        "log": log_path,
    }
    if config.save_ocr_dump and cached_ocr is None:
        outputs["ocr_dump"] = ocr_dump
    if print_summary:
        print_console_summary(source, tracked, errors, outputs)
    stats.elapsed_s = time.perf_counter() - started
    accounted = stats.change_s + stats.ocr_s + stats.classify_s + stats.tracking_s + stats.render_s
    stats.decode_s = max(0.0, stats.elapsed_s - accounted)
    logger.info("Finished LeakLens run. Original video unchanged.")
    return ScanResult(
        source=source,
        video_info=info,
        detections=tracked,
        errors=errors,
        outputs=outputs,
        summary=summarize(tracked),
        performance=stats,
    )


def _ocr_for_change(ocr, frame, change, prev_items, cache) -> tuple[list[OcrItem], list[str], bool]:
    if change.full_frame or prev_items is None:
        items, errors = ocr.run(frame)
        return items, errors, True
    regions = _compact_ocr_regions(list(change.regions), int(frame.shape[0] * frame.shape[1]))
    new_items: list[OcrItem] = []
    collected: list[str] = []
    for region in regions:
        x1, y1, x2, y2 = region.as_xyxy()
        crop = frame[y1:y2, x1:x2]
        region_items, region_errors = ocr.run_region(crop, region, cache=cache)
        new_items.extend(region_items)
        collected.extend(region_errors)
    kept = [item for item in prev_items if not item_overlaps_regions(item.bbox, change.regions)]
    return collapse_overlapping_ocr(kept + new_items), collected, False


def _classify_frame(
    frame,
    frame_number,
    timestamp,
    info,
    config,
    items,
    ocr_errors,
    regex,
    heuristic,
    extra,
    visual,
    *,
    run_visual: bool,
    reused_visual: list[Detection] | None = None,
) -> tuple[list[Detection], list[Detection]]:
    ctx = FrameContext(
        frame_bgr=frame,
        frame_number=frame_number,
        timestamp=timestamp,
        width=info.width,
        height=info.height,
        ocr_items=items,
        ocr_errors=ocr_errors,
        config=config,
    )
    text_dets = regex.detect(ctx) + heuristic.detect(ctx) + extra.detect(ctx)
    if run_visual:
        visual_dets = visual.detect(ctx)
    else:
        visual_dets = _restamp(reused_visual or [], frame_number, timestamp)
    return text_dets, visual_dets


def _omnibox_region(frame_w: int, frame_h: int) -> BBox:
    """Chrome address bar. Re-OCR this so sslip IPs are not dropped after navigation."""
    return BBox(
        max(0, int(frame_w * 0.03)),
        max(0, int(frame_h * 0.032)),
        min(frame_w, int(frame_w * 0.58)),
        max(40, int(frame_h * 0.052)),
    )


def _compact_ocr_regions(regions: list[BBox], frame_area: int) -> list[BBox]:
    if len(regions) <= 1:
        return regions
    union = union_boxes(regions)
    if union is None:
        return regions
    if union.w * union.h > int(0.45 * frame_area):
        return regions
    top = min(r.y for r in regions)
    bottom = max(r.y + r.h for r in regions)
    if bottom - top > 140:
        return regions
    return [union]


def _has_omnibox_url(items: list[OcrItem] | None, frame_w: int, frame_h: int) -> bool:
    if not items:
        return False
    omni = _omnibox_region(frame_w, frame_h)
    omni_bottom = omni.y + omni.h
    for item in items:
        text = (item.text or "").lower()
        if not text:
            continue
        if item.bbox.y > omni_bottom or item.bbox.y + item.bbox.h < omni.y:
            continue
        if any(token in text for token in ("http", "www.", "sslip", "nip.io", "xip.io")):
            return True
        if sum(ch == "." for ch in text) >= 3 and any(ch.isdigit() for ch in text):
            return True
    return False


def _sticky_scan_regions(
    dets: list[Detection],
    frame_w: int,
    frame_h: int,
    *,
    include_omnibox: bool,
) -> list[BBox]:
    regions: list[BBox] = []
    if include_omnibox:
        regions.append(_omnibox_region(frame_w, frame_h))
    chrome_y = int(frame_h * 0.14)
    boxes = [
        d.bbox
        for d in dets
        if d.category in {"autofill_suggestion", "email"} and d.bbox.y > chrome_y
    ]
    if boxes:
        union = union_boxes(boxes)
        if union is not None:
            grown = union.padded(48, frame_w, frame_h)
            lift = 72
            y = max(chrome_y, grown.y - lift)
            regions.append(BBox(grown.x, y, grown.w, grown.y + grown.h - y))
    return regions


def _restamp(dets: list[Detection], frame_number: int, timestamp: float) -> list[Detection]:
    return [
        replace(
            det,
            frame_number=frame_number,
            timestamp=timestamp,
            origin="direct",
            temporal_extended=False,
        )
        for det in dets
    ]


def _configure_logging(log_path: Path, *, console: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if console else logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [file_handler]
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        handlers.append(stream_handler)
    root.handlers = handlers
