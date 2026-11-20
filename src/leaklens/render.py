from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from leaklens.classify import detection_should_redact
from leaklens.config import AppConfig
from leaklens.exceptions import ExportError
from leaklens.models import BBox, Detection
from leaklens.video_io import is_image_path, read_bgr, write_bgr

logger = logging.getLogger(__name__)


def partial_output_path(destination: Path) -> Path:
    """Temp encode path that still has a real video extension (ffmpeg muxer)."""
    return destination.with_name(f"{destination.stem}.partial{destination.suffix}")


def render_redacted_media(
    source: Path,
    destination: Path,
    detections: list[Detection],
    config: AppConfig,
    fps: float,
    *,
    on_frame: Callable[[int, int], None] | None = None,
    frame_count: int | None = None,
) -> None:
    if is_image_path(source):
        render_redacted_image(source, destination, detections, config, on_frame=on_frame)
        return
    render_redacted_video(
        source,
        destination,
        detections,
        config,
        fps,
        on_frame=on_frame,
        frame_count=frame_count,
    )


def render_redacted_image(
    source: Path,
    destination: Path,
    detections: list[Detection],
    config: AppConfig,
    *,
    on_frame: Callable[[int, int], None] | None = None,
) -> None:
    frame = read_bgr(source)
    boxes = _merge_boxes([det.bbox for det in detections if detection_should_redact(det, config)])
    out = apply_blur(frame, boxes, config)
    try:
        write_bgr(destination, out)
    except Exception as exc:
        raise ExportError(f"Could not write the redacted image: {exc}") from exc
    if on_frame:
        on_frame(1, 1)


def render_redacted_video(
    source: Path,
    destination: Path,
    detections: list[Detection],
    config: AppConfig,
    fps: float,
    *,
    on_frame: Callable[[int, int], None] | None = None,
    frame_count: int | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[BBox]] = {}
    for det in detections:
        if detection_should_redact(det, config):
            by_frame.setdefault(det.frame_number, []).append(det.bbox)

    probe = cv2.VideoCapture(str(source))
    if not probe.isOpened():
        raise ExportError(f"Could not reopen video for rendering: {source}")
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = frame_count or int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    probe.release()

    partial = partial_output_path(destination)
    if partial.exists():
        partial.unlink()

    ffmpeg = shutil.which("ffmpeg")
    last_error: Exception | None = None
    if ffmpeg:
        try:
            _render_pass(
                source,
                partial,
                by_frame,
                config,
                fps,
                width,
                height,
                total,
                on_frame,
                _FfmpegPipeEncoder(partial, source, fps, width, height),
            )
            os.replace(partial, destination)
            return
        except Exception as exc:
            last_error = exc
            logger.warning("ffmpeg encode failed (%s); falling back to OpenCV", exc)
            if partial.exists():
                partial.unlink(missing_ok=True)

    logger.warning("Writing OpenCV mp4v preview without original audio")
    try:
        _render_pass(
            source,
            partial,
            by_frame,
            config,
            fps,
            width,
            height,
            total,
            on_frame,
            _OpenCvEncoder(partial, fps, width, height),
        )
        os.replace(partial, destination)
    except Exception as exc:
        if partial.exists():
            partial.unlink(missing_ok=True)
        detail = last_error or exc
        raise ExportError(f"Could not encode the redacted video: {detail}") from exc


def _render_pass(
    source: Path,
    partial: Path,
    by_frame: dict[int, list[BBox]],
    config: AppConfig,
    fps: float,
    width: int,
    height: int,
    total: int | None,
    on_frame: Callable[[int, int], None] | None,
    encoder: _FrameEncoder,
) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        encoder.close(abort=True)
        raise ExportError(f"Could not reopen video for rendering: {source}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = _merge_boxes(by_frame.get(index, []))
            encoder.write(apply_blur(frame, boxes, config))
            index += 1
            if on_frame:
                on_frame(index, total or index)
        encoder.close()
        if not partial.exists() or partial.stat().st_size < 32:
            raise ExportError("encoder produced an empty redacted video")
    except BaseException:
        encoder.close(abort=True)
        raise
    finally:
        cap.release()


class _FrameEncoder:
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self, abort: bool = False) -> None:
        raise NotImplementedError


class _OpenCvEncoder(_FrameEncoder):
    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        if path.exists():
            path.unlink()
        self._writer = _opencv_writer(path, fps, width, height)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.release()


class _FfmpegPipeEncoder(_FrameEncoder):
    def __init__(self, destination: Path, source: Path, fps: float, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._stderr = b""
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps or 30),
            "-i",
            "-",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(destination),
        ]
        logger.info("Encoding redacted preview with ffmpeg pipe")
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._closed = False
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,
            creationflags=flags,
        )
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._err_thread.start()

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        self._stderr = self._proc.stderr.read() or b""

    def write(self, frame: np.ndarray) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(self._error_text() or "ffmpeg exited early")
        assert self._proc.stdin is not None
        blob = np.ascontiguousarray(frame)
        if blob.shape[1] != self.width or blob.shape[0] != self.height:
            raise RuntimeError("frame size changed during encode")
        self._proc.stdin.write(blob.tobytes())

    def close(self, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        if abort:
            self._proc.kill()
        try:
            code = self._proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)
            code = -1
        self._err_thread.join(timeout=5)
        if self._proc.stderr:
            try:
                self._proc.stderr.close()
            except OSError:
                pass
        if abort:
            return
        if code != 0:
            raise ExportError(f"ffmpeg could not encode the redacted video: {self._error_text()}")

    def _error_text(self) -> str:
        text = self._stderr.decode("utf-8", errors="replace").strip()
        return text[-1500:] if text else "unknown ffmpeg error"


def apply_blur(frame: np.ndarray, boxes: list[BBox], config: AppConfig) -> np.ndarray:
    if not boxes:
        return frame
    out = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.as_xyxy()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(out.shape[1], x2)
        y2 = min(out.shape[0], y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        height, width = roi.shape[:2]
        if height >= 48 or width >= 160:
            small = cv2.resize(
                roi,
                (max(2, width // 36), max(2, height // 36)),
                interpolation=cv2.INTER_AREA,
            )
            out[y1:y2, x1:x2] = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            continue
        # Thin URL/email glyphs stay readable with 6-block mosaics; crush to 2-3 slabs.
        blocks_x = 2 if width < 90 else 3
        blocks_y = 1 if height <= 28 else 2
        small = cv2.resize(roi, (blocks_x, blocks_y), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return out


def _merge_boxes(boxes: list[BBox]) -> list[BBox]:
    """Merge near-duplicate boxes only. Adjacent secrets on one line stay separate."""
    remaining = list(boxes)
    merged: list[BBox] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[BBox] = []
            for other in remaining:
                if current.iou(other) >= 0.45:
                    current = current.union(other)
                    changed = True
                else:
                    keep.append(other)
            remaining = keep
        merged.append(current)
    return merged


def _opencv_writer(path: Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps or 30.0, (width, height))
    if not writer.isOpened():
        raise ExportError(f"Could not open VideoWriter for {path}")
    return writer
