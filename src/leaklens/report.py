from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from leaklens.models import BBox, Detection, OcrItem


def write_detections_json(
    path: Path,
    *,
    source: Path,
    video_meta: dict[str, Any],
    config: dict[str, Any],
    detections: list[Detection],
    errors: list[str],
    ocr_stats: dict[str, Any],
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prototype": True,
        "production_ready": False,
        "source_video": str(source),
        "source_preserved": True,
        "video": video_meta,
        "config": config,
        "errors": errors,
        "ocr_stats": ocr_stats,
        "summary": summarize(detections),
        "detections": [det.to_dict() for det in detections],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_ocr_dump(path: Path, frames: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frames:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_ocr_dump(path: Path) -> dict[int, list[OcrItem]]:
    frames: dict[int, list[OcrItem]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            items = []
            for item in row.get("items") or []:
                box = item.get("bbox") or {}
                items.append(
                    OcrItem(
                        text=item.get("text") or "",
                        confidence=float(item.get("confidence") or 0.0),
                        bbox=BBox(
                            x=int(box.get("x") or 0),
                            y=int(box.get("y") or 0),
                            w=int(box.get("w") or 1),
                            h=int(box.get("h") or 1),
                        ),
                        source=item.get("source") or "dump",
                        inverted=bool(item.get("inverted")),
                    )
                )
            frames[int(row["frame_number"])] = items
    return frames


def summarize(detections: list[Detection]) -> dict[str, Any]:
    by_category: dict[str, int] = Counter(d.category for d in detections)
    by_class: dict[str, int] = Counter(d.classification for d in detections)
    by_origin: dict[str, int] = Counter(d.origin for d in detections)
    tracks = {d.track_id for d in detections if d.track_id is not None}
    return {
        "detection_count": len(detections),
        "track_count": len(tracks),
        "review_required": sum(1 for d in detections if d.classification == "REVIEW_REQUIRED"),
        "sensitive": sum(1 for d in detections if d.classification == "SENSITIVE"),
        "temporal_extended": sum(1 for d in detections if d.temporal_extended),
        "by_classification": dict(sorted(by_class.items())),
        "by_origin": dict(sorted(by_origin.items())),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def print_console_summary(
    source: Path,
    detections: list[Detection],
    errors: list[str],
    outputs: dict[str, Path],
) -> None:
    summary = summarize(detections)
    print("\n=== Leak detection summary (prototype, not production-ready) ===")
    print(f"Source: {source}")
    print(f"Detections: {summary['detection_count']}  tracks: {summary['track_count']}")
    print(f"SENSITIVE: {summary.get('sensitive', 0)}  REVIEW_REQUIRED: {summary['review_required']}")
    print(f"Temporal extensions: {summary['temporal_extended']}")
    if summary.get("by_origin"):
        print("By origin:", ", ".join(f"{k}={v}" for k, v in summary["by_origin"].items()))
    print("By category:")
    if summary["by_category"]:
        width = max(len(name) for name in summary["by_category"])
        for name, count in summary["by_category"].items():
            print(f"  {name.ljust(width)}  {count}")
    else:
        print("  (none)")
    if errors:
        print(f"Logged issues: {len(errors)}")
        for err in errors[:12]:
            print(f"  - {err}")
        if len(errors) > 12:
            print(f"  ... {len(errors) - 12} more")
    print("Wrote:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    print("Original video was not modified.")


def write_review_html(
    path: Path,
    *,
    source: Path,
    detections: list[Detection],
    crops_dir: Path,
    frames_by_number: dict[int, Any],
    max_crops: int,
) -> None:
    tracks = representative_tracks(detections)
    crops_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    saved = 0
    for track_id, det in tracks:
        crop_rel = ""
        frame = frames_by_number.get(det.frame_number)
        if frame is not None and saved < max_crops:
            crop_name = f"track_{track_id:04d}.jpg"
            crop_path = crops_dir / crop_name
            _save_crop(frame, det, crop_path)
            crop_rel = f"crops/{crop_name}"
            saved += 1
        cards.append(_card_html(track_id, det, crop_rel, detections))

    summary = summarize(detections)
    category_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in summary["by_category"].items()
    )
    origin_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in (summary.get("by_origin") or {}).items()
    )
    detection_rows = "".join(_decision_row_html(det) for det in detections)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Leak review — {html.escape(source.name)}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; background: #101216; color: #e8eaed; }}
    a {{ color: #8ab4f8; }}
    .meta, table {{ margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #2a2f3a; padding: 6px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ background: #1a1f29; border: 1px solid #2c3340; border-radius: 12px; padding: 12px; }}
    .card img {{ width: 100%; border-radius: 8px; background: #000; }}
    .tag {{ display: inline-block; background: #3b2a12; color: #ffb74d; padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
    .ok {{ background: #14331f; color: #81c995; }}
    code {{ font-size: 12px; word-break: break-all; }}
    .warn {{ background: #3a2410; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; }}
    #filter {{ width: min(480px, 100%); padding: 8px 10px; margin: 8px 0 16px; background: #1a1f29; color: #e8eaed; border: 1px solid #2c3340; border-radius: 8px; }}
    .decisions {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Leak detection review</h1>
  <div class="warn">Prototype only. OCR alone does not cause a blur. Every row below is a classified secret span, including temporally held frames. Not production-ready.</div>
  <div class="meta">
    <div>Source: <code>{html.escape(str(source))}</code></div>
    <div>Detections: {summary['detection_count']} · Tracks: {summary['track_count']} · SENSITIVE: {summary.get('sensitive', 0)} · REVIEW_REQUIRED: {summary['review_required']} · Temporal extensions: {summary.get('temporal_extended', 0)}</div>
  </div>
  <h2>By category</h2>
  <table>
    <thead><tr><th>Category</th><th>Count</th></tr></thead>
    <tbody>{category_rows or '<tr><td colspan="2">None</td></tr>'}</tbody>
  </table>
  <h2>By origin</h2>
  <table>
    <thead><tr><th>Origin</th><th>Count</th></tr></thead>
    <tbody>{origin_rows or '<tr><td colspan="2">None</td></tr>'}</tbody>
  </table>
  <h2>Detection tracks</h2>
  <div class="grid">
    {''.join(cards) or '<p>No detections.</p>'}
  </div>
  <h2>Why every blur exists</h2>
  <p>Filter by frame, text, detector, classification, or origin.</p>
  <input id="filter" type="search" placeholder="e.g. ipv4, temporal_hold, password, frame 98">
  <div class="decisions">
  <table>
    <thead><tr>
      <th>t</th><th>frame</th><th>class</th><th>category</th><th>detector</th><th>origin</th>
      <th>OCR text</th><th>normalized</th><th>conf</th><th>bbox</th><th>first</th><th>last</th><th>track</th>
    </tr></thead>
    <tbody id="decisions">{detection_rows or '<tr><td colspan="13">None</td></tr>'}</tbody>
  </table>
  </div>
  <script>
    const input = document.getElementById('filter');
    const rows = [...document.querySelectorAll('#decisions tr')];
    input.addEventListener('input', () => {{
      const q = input.value.toLowerCase();
      rows.forEach(row => {{
        row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def ocr_items_payload(items: list[OcrItem]) -> list[dict[str, Any]]:
    return [
        {
            "text": item.text,
            "confidence": round(item.confidence, 4),
            "bbox": item.bbox.to_dict(),
            "source": item.source,
            "inverted": item.inverted,
        }
        for item in items
    ]


def representative_tracks(detections: list[Detection]) -> list[tuple[int, Detection]]:
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        grouped[det.track_id or 0].append(det)
    reps: list[tuple[int, Detection]] = []
    for track_id, group in grouped.items():
        original = [d for d in group if not d.temporal_extended] or group
        original.sort(key=lambda d: (-d.confidence, d.frame_number))
        reps.append((track_id, original[0]))
    reps.sort(key=lambda pair: (pair[1].frame_number, pair[0]))
    return reps


def _save_crop(frame, det: Detection, path: Path) -> None:
    x1, y1, x2, y2 = det.bbox.as_xyxy()
    pad = 8
    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(frame.shape[0], y2 + pad)
    x2 = min(frame.shape[1], x2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    cv2.imwrite(str(path), crop)


def _card_html(track_id: int, det: Detection, crop_rel: str, all_dets: list[Detection]) -> str:
    span = [d for d in all_dets if d.track_id == det.track_id]
    frames = sorted({d.frame_number for d in span})
    review = (
        '<span class="tag">REVIEW_REQUIRED</span>'
        if det.classification == "REVIEW_REQUIRED"
        else '<span class="tag ok">SENSITIVE</span>'
    )
    origin = "temporally propagated" if det.temporal_extended else "direct"
    img = f'<img src="{html.escape(crop_rel)}" alt="crop">' if crop_rel else "<em>No crop</em>"
    text = html.escape(det.text) if det.text else "<em>n/a</em>"
    norm = html.escape(det.normalized_text or "")
    return f"""
    <article class="card">
      {img}
      <p>{review} <strong>{html.escape(det.category)}</strong></p>
      <p>track {track_id} · first frame {det.first_frame_detected if det.first_frame_detected is not None else frames[0]} · last redacted {det.last_frame_redacted if det.last_frame_redacted is not None else frames[-1]}</p>
      <p>shown frame {det.frame_number} · t={det.timestamp:.2f}s · {html.escape(origin)}</p>
      <p>detector: <code>{html.escape(det.detector)}</code> · conf {det.confidence:.2f}</p>
      <p>OCR text: <code>{text}</code></p>
      <p>normalized: <code>{norm}</code></p>
      <p>bbox: {det.bbox.x},{det.bbox.y} {det.bbox.w}x{det.bbox.h}</p>
      <p>frames: {frames[0]}–{frames[-1]} ({len(frames)})</p>
    </article>
    """


def _decision_row_html(det: Detection) -> str:
    origin = det.origin or ("temporal" if det.temporal_extended else "direct")
    bbox = f"{det.bbox.x},{det.bbox.y} {det.bbox.w}x{det.bbox.h}"
    return (
        "<tr>"
        f"<td>{det.timestamp:.2f}</td>"
        f"<td>{det.frame_number}</td>"
        f"<td>{html.escape(det.classification)}</td>"
        f"<td>{html.escape(det.category)}</td>"
        f"<td><code>{html.escape(det.detector)}</code></td>"
        f"<td>{html.escape(origin)}</td>"
        f"<td><code>{html.escape(det.text or '')}</code></td>"
        f"<td><code>{html.escape(det.normalized_text or '')}</code></td>"
        f"<td>{det.confidence:.2f}</td>"
        f"<td>{html.escape(bbox)}</td>"
        f"<td>{det.first_frame_detected if det.first_frame_detected is not None else ''}</td>"
        f"<td>{det.last_frame_redacted if det.last_frame_redacted is not None else ''}</td>"
        f"<td>{det.track_id if det.track_id is not None else ''}</td>"
        "</tr>"
    )
