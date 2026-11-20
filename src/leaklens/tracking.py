from __future__ import annotations

from collections import defaultdict

from leaklens.config import AppConfig
from leaklens.detectors.regex import is_ascii_art
from leaklens.models import BBox, Detection, OcrItem
from leaklens.util import normalize_secret


def apply_temporal_tracking(
    detections: list[Detection],
    config: AppConfig,
    fps: float,
    *,
    ocr_by_frame: dict[int, list[OcrItem]] | None = None,
    frame_count: int | None = None,
) -> list[Detection]:
    """Link per-frame secrets and hold redaction through motion-blurred frames.

    Every direct detection is kept, including single-frame leaks. Neighboring
    frames are examined to extend the box, never to skip analysis.
    """
    if not detections:
        return []

    merged = merge_frame_detections(detections, iou_threshold=max(0.2, config.track_iou_threshold))
    ordered = sorted(merged, key=lambda d: (d.frame_number, d.category, d.bbox.y, d.bbox.x))
    tracks: list[list[Detection]] = []

    for det in ordered:
        assigned = None
        for track in reversed(tracks):
            last = track[-1]
            gap = det.frame_number - last.frame_number
            if gap < 0 or gap > config.track_max_gap_frames:
                continue
            if not _same_secret(last, det):
                continue
            iou = last.bbox.iou(det.bbox)
            if gap == 0 and iou < 0.45:
                continue
            if iou < config.track_iou_threshold and not _near_same_row(last, det):
                continue
            assigned = track
            break
        if assigned is None:
            tracks.append([det])
        else:
            assigned.append(det)

    tracked: list[Detection] = []
    max_frame = (frame_count - 1) if frame_count else max(d.frame_number for d in merged)
    for idx, track in enumerate(tracks, start=1):
        expanded = _expand_track(track, config, fps, ocr_by_frame or {}, max_frame)
        first_detected = min(d.frame_number for d in track)
        last_redacted = max(d.frame_number for d in expanded)
        for det in expanded:
            det.track_id = idx
            det.first_frame_detected = first_detected
            det.last_frame_redacted = last_redacted
            tracked.append(det)

    return sorted(tracked, key=lambda d: (d.frame_number, d.track_id or 0))


def merge_frame_detections(detections: list[Detection], iou_threshold: float = 0.3) -> list[Detection]:
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        grouped[det.frame_number].append(det)
    merged: list[Detection] = []
    for group in grouped.values():
        kept: list[Detection] = []
        for det in sorted(group, key=lambda d: -d.confidence):
            target = None
            for existing in kept:
                if not _same_secret(existing, det):
                    continue
                if existing.bbox.iou(det.bbox) >= iou_threshold:
                    target = existing
                    break
            if target is None:
                kept.append(det)
                continue
            target.bbox = target.bbox.union(det.bbox)
            if det.confidence > target.confidence:
                target.confidence = det.confidence
                target.detector = det.detector
            target.review_required = target.review_required or det.review_required
            if det.classification == "REVIEW_REQUIRED":
                target.classification = "REVIEW_REQUIRED"
            target.extra.update(det.extra)
        merged.extend(kept)
    return merged


def _expand_track(
    track: list[Detection],
    config: AppConfig,
    fps: float,
    ocr_by_frame: dict[int, list[OcrItem]],
    max_frame: int,
) -> list[Detection]:
    by_frame = {det.frame_number: det for det in sorted(track, key=lambda d: d.frame_number)}
    first_direct = min(by_frame)
    last_direct = max(by_frame)
    source = by_frame[first_direct]

    if config.track_extend_unmatched:
        for frame_no in range(first_direct, last_direct + 1):
            if frame_no in by_frame:
                continue
            nearest = min(by_frame, key=lambda f: abs(f - frame_no))
            timestamp = frame_no / fps if fps else source.timestamp
            clone = by_frame[nearest].clone_on_frame(frame_no, timestamp, origin="temporal_gap")
            clone.bbox = by_frame[nearest].bbox
            by_frame[frame_no] = clone

        hold = max(config.track_max_gap_frames, config.track_hold_unreadable_frames)
        for frame_no in range(last_direct + 1, min(max_frame, last_direct + hold) + 1):
            if not _should_hold(by_frame[frame_no - 1], ocr_by_frame.get(frame_no, [])):
                break
            timestamp = frame_no / fps if fps else source.timestamp
            clone = by_frame[frame_no - 1].clone_on_frame(frame_no, timestamp, origin="temporal_hold")
            by_frame[frame_no] = clone
        for frame_no in range(first_direct - 1, max(-1, first_direct - hold - 1), -1):
            if frame_no < 0:
                break
            if not _should_hold(by_frame[frame_no + 1], ocr_by_frame.get(frame_no, [])):
                break
            timestamp = frame_no / fps if fps else source.timestamp
            clone = by_frame[frame_no + 1].clone_on_frame(frame_no, timestamp, origin="temporal_hold")
            by_frame[frame_no] = clone

    return list(by_frame.values())


def _should_hold(previous: Detection, items: list[OcrItem]) -> bool:
    """Keep blurring if the region is unreadable or still contains the secret."""
    secret = previous.normalized_text or normalize_secret(previous.text)
    overlapping = [item for item in items if previous.bbox.iou(item.bbox) > 0.05 or _contains(previous.bbox, item.bbox)]
    if not overlapping:
        return True
    texts = [item.text for item in overlapping if item.text]
    if any(secret and secret in normalize_secret(text) for text in texts):
        return True
    confident = [item for item in overlapping if item.confidence >= 0.55 and item.text and not is_ascii_art(item.text)]
    if not confident:
        return True
    joined = normalize_secret(" ".join(item.text for item in confident))
    return bool(secret) and secret in joined


def _same_secret(a: Detection, b: Detection) -> bool:
    if a.category != b.category and (a.category, b.category) not in {
        ("ipv4_private", "ipv4_public"),
        ("ipv4_public", "ipv4_private"),
        ("hostname", "username_handle"),
        ("username_handle", "hostname"),
    }:
        return False
    left = a.normalized_text or normalize_secret(a.text)
    right = b.normalized_text or normalize_secret(b.text)
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return len(shorter) >= 6 and shorter in longer


def _near_same_row(a: Detection, b: Detection) -> bool:
    return abs((a.bbox.y + a.bbox.h / 2) - (b.bbox.y + b.bbox.h / 2)) <= 18 and _same_secret(a, b)


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x + inner.w <= outer.x + outer.w
        and inner.y + inner.h <= outer.y + outer.h
    )
