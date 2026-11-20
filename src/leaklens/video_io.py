from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}

VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".wmv",
    ".mpeg",
    ".mpg",
    ".m4s",
    ".ts",
}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def read_bgr(path: Path):
    """Read an image as BGR, including Windows paths cv2.imread cannot open."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"Could not read image: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unsupported or empty image: {path}")
    return image


def write_bgr(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(suffix if suffix in IMAGE_SUFFIXES else ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buf.tofile(str(path))


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3),
        }


def probe_video(path: Path) -> VideoInfo:
    if is_image_path(path):
        image = read_bgr(path)
        height, width = image.shape[:2]
        return VideoInfo(
            path=path,
            width=int(width),
            height=int(height),
            fps=1.0,
            frame_count=1,
            duration_s=0.0,
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return VideoInfo(path=path, width=width, height=height, fps=fps, frame_count=frame_count, duration_s=duration)


def iter_frames(path: Path, stride: int = 1):
    if is_image_path(path):
        yield 0, 0.0, read_bgr(path)
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    index = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % max(1, stride) == 0:
                yield index, index / fps, frame
            index += 1
    finally:
        cap.release()


def grab_frames(path: Path, frame_numbers: set[int]) -> dict[int, object]:
    wanted = set(frame_numbers)
    grabbed: dict[int, object] = {}
    if not wanted:
        return grabbed
    if is_image_path(path):
        if 0 in wanted:
            grabbed[0] = read_bgr(path)
        return grabbed
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {path}")
    index = 0
    try:
        while wanted and index <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                grabbed[index] = frame.copy()
                wanted.remove(index)
            index += 1
    finally:
        cap.release()
    return grabbed
