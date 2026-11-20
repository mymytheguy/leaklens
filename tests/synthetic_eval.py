"""Synthetic leak videos + recall/false-positive scoring.

This is a verification harness, not a claim of 100% accuracy.
Run: python tests/synthetic_eval.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaklens.config import AppConfig
from leaklens.models import BBox, OcrItem
from leaklens.pipeline import run_pipeline
from leaklens.report import ocr_items_payload, write_ocr_dump


def _blank(w: int = 640, h: int = 180) -> np.ndarray:
    return np.full((h, w, 3), 20, dtype=np.uint8)


def _put(frame: np.ndarray, text: str, x: int, y: int, scale: float = 0.55) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 90), 1, cv2.LINE_AA)


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _item(text: str, x: int, y: int, w: int, h: int = 18, conf: float = 0.95) -> dict:
    return ocr_items_payload([OcrItem(text, conf, BBox(x, y, w, h))])[0]


def _dump_row(frame_number: int, items: list[dict]) -> dict:
    return {"frame_number": frame_number, "timestamp": frame_number / 30, "errors": [], "items": items}


def build_cases(tmp: Path) -> list[dict]:
    cases: list[dict] = []

    frames, dump, gts = [], [], []
    for i in range(12):
        frame = _blank()
        if i == 5:
            _put(frame, "10.9.8.7", 40, 90)
            items = [_item("10.9.8.7", 40, 74, 110)]
            gts.append({"frame": i, "text": "10.9.8.7", "bbox": {"x": 40, "y": 74, "w": 110, "h": 18}})
        else:
            _put(frame, "apt update && true", 40, 90)
            items = [_item("apt update && true", 40, 74, 260)]
        frames.append(frame)
        dump.append(_dump_row(i, items))
    video, ocr = tmp / "one_frame.mp4", tmp / "one_frame.jsonl"
    _write_video(video, frames)
    write_ocr_dump(ocr, dump)
    cases.append({"name": "one_frame", "video": video, "ocr": ocr, "gts": gts, "safe_words": ["apt", "update"]})

    frames, dump, gts = [], [], []
    for i in range(10):
        frame = _blank()
        y = 40 + i * 10
        _put(frame, "192.168.10.12", 30, y)
        items = [_item("192.168.10.12", 30, y - 16, 160)]
        frames.append(frame)
        dump.append(_dump_row(i, items))
        gts.append({"frame": i, "text": "192.168.10.12", "bbox": {"x": 30, "y": y - 16, "w": 160, "h": 18}})
    video, ocr = tmp / "scrolling.mp4", tmp / "scrolling.jsonl"
    _write_video(video, frames)
    write_ocr_dump(ocr, dump)
    cases.append({"name": "scrolling", "video": video, "ocr": ocr, "gts": gts, "safe_words": []})

    frames, dump, gts = [], [], []
    line = "curl https://192.168.1.50/api"
    for i in range(8):
        frame = _blank()
        _put(frame, line, 20, 80)
        frames.append(frame)
        dump.append(_dump_row(i, [_item(line, 20, 64, 360)]))
        gts.append({"frame": i, "text": "192.168.1.50", "bbox": {"x": 140, "y": 64, "w": 120, "h": 18}})
    video, ocr = tmp / "beside_ordinary.mp4", tmp / "beside_ordinary.jsonl"
    _write_video(video, frames)
    write_ocr_dump(ocr, dump)
    cases.append({"name": "beside_ordinary", "video": video, "ocr": ocr, "gts": gts, "safe_words": ["curl"], "curl_max_x": 80})

    frames, dump, gts = [], [], []
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.aaaaaaaaaaaaaaaaaaaaaa"
    for i in range(15):
        frame = _blank()
        if i == 0:
            _put(frame, "token=" + jwt, 10, 90, 0.35)
            items = [_item(jwt, 70, 74, 400)]
            gts.append({"frame": 0, "text": "eyJ", "bbox": {"x": 70, "y": 74, "w": 400, "h": 18}})
        else:
            _put(frame, "npm run build", 10, 90)
            items = [_item("npm run build", 10, 74, 180)]
        frames.append(frame)
        dump.append(_dump_row(i, items))
    video, ocr = tmp / "few_ms.mp4", tmp / "few_ms.jsonl"
    _write_video(video, frames)
    write_ocr_dump(ocr, dump)
    cases.append({"name": "few_milliseconds", "video": video, "ocr": ocr, "gts": gts, "safe_words": ["npm", "run", "build"]})

    frames, dump, gts = [], [], []
    for i in range(8):
        frame = _blank()
        x = 20 + i * 18
        _put(frame, "8.8.8.8", x, 90, 0.4)
        items = [_item("8.8.8.8", x, 74, 70)]
        frames.append(frame)
        dump.append(_dump_row(i, items))
        gts.append({"frame": i, "text": "8.8.8.8", "bbox": {"x": x, "y": 74, "w": 70, "h": 18}})
    video, ocr = tmp / "moving.mp4", tmp / "moving.jsonl"
    _write_video(video, frames)
    write_ocr_dump(ocr, dump)
    cases.append({"name": "moving", "video": video, "ocr": ocr, "gts": gts, "safe_words": []})

    frames, dump, gts = [], [], []
    for i in range(6):
        frame = _blank()
        _put(frame, "10.0.0.9", 40, 90, 0.35)
        items = [_item("10.0.0.9", 40, 78, 70, 12, conf=0.4)]
        frames.append(frame)
        dump.append(_dump_row(i, items))
        gts.append({"frame": i, "text": "10.0.0.9", "bbox": {"x": 40, "y": 78, "w": 70, "h": 12}})
    video, ocr = tmp / "small_font.mp4", tmp / "small_font.jsonl"
    _write_video(video, frames)
    write_ocr_dump(ocr, dump)
    cases.append({"name": "small_font", "video": video, "ocr": ocr, "gts": gts, "safe_words": []})
    return cases


def _iou(gt_box: dict, det_box: dict) -> float:
    ax1, ay1, ax2, ay2 = gt_box["x"], gt_box["y"], gt_box["x"] + gt_box["w"], gt_box["y"] + gt_box["h"]
    bx1, by1, bx2, by2 = det_box["x"], det_box["y"], det_box["x"] + det_box["w"], det_box["y"] + det_box["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = gt_box["w"] * gt_box["h"] + det_box["w"] * det_box["h"] - inter
    return inter / union if union else 0.0


def evaluate(case: dict, detections: list[dict]) -> dict:
    gts = case["gts"]
    hits = 0
    for gt in gts:
        frame_dets = [d for d in detections if d["frame_number"] == gt["frame"]]
        if any(
            _iou(gt["bbox"], d["bbox"]) > 0.05 or gt["text"].lower() in (d.get("normalized_text") or d.get("text") or "").lower()
            for d in frame_dets
        ):
            hits += 1
    recall = hits / len(gts) if gts else 1.0
    fp = 0
    for det in detections:
        text = (det.get("text") or "").strip().lower()
        if text in {w.lower() for w in case.get("safe_words") or []}:
            fp += 1
        if case.get("curl_max_x") is not None and det["bbox"]["x"] + det["bbox"]["w"] < case["curl_max_x"]:
            fp += 1
    return {"name": case["name"], "recall": recall, "false_positives": fp, "detections": len(detections)}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="leaklens_synth_"))
    results = []
    for case in build_cases(tmp):
        out = tmp / f"out_{case['name']}"
        run_pipeline(
            case["video"],
            out,
            AppConfig(ocr_run_inverted=False, ocr_run_sharpen=False, save_ocr_dump=False),
            ocr_dump_path=case["ocr"],
        )
        payload = json.loads((out / "detections.json").read_text(encoding="utf-8"))
        score = evaluate(case, payload["detections"])
        results.append(score)
        print(f"{score['name']}: recall={score['recall']:.2f} false_positives={score['false_positives']} dets={score['detections']}")
    misses = [r for r in results if r["recall"] < 1.0]
    print("synthetic dir:", tmp)
    if misses:
        print("MISS: sensitive recall is not 1.0 on", [m["name"] for m in misses])
        return 1
    print("Sensitive-information recall was 1.0 on the synthetic suite. Not production-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
