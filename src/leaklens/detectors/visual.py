from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from leaklens.config import AppConfig
from leaklens.detectors.base import FrameContext
from leaklens.models import BBox, Detection

logger = logging.getLogger(__name__)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


class VisualDetector:
    name = "visual"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._face = None
        self._face_error: str | None = None
        self._qr = cv2.QRCodeDetector()
        self._barcode = None
        try:
            self._barcode = cv2.barcode.BarcodeDetector()
        except Exception as exc:  # pragma: no cover - optional OpenCV module
            logger.warning("Barcode detector unavailable: %s", exc)

    def detect(self, ctx: FrameContext) -> list[Detection]:
        detections: list[Detection] = []
        detections.extend(self._detect_faces(ctx))
        detections.extend(self._detect_qr(ctx))
        detections.extend(self._detect_barcodes(ctx))
        detections.extend(self._detect_plate_like(ctx))
        return detections

    def _face_model(self, width: int, height: int):
        if self._face_error:
            return None
        if self._face is not None:
            try:
                self._face.setInputSize((width, height))
            except Exception:
                pass
            return self._face
        model_path = ASSET_DIR / "face_detection_yunet_2023mar.onnx"
        try:
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            if not model_path.exists() or model_path.stat().st_size < 1000:
                logger.info("Downloading local YuNet face model (one-time)")
                urllib.request.urlretrieve(YUNET_URL, model_path)
            self._face = cv2.FaceDetectorYN_create(  # type: ignore[attr-defined]
                str(model_path),
                "",
                (width, height),
                0.6,
                0.3,
                5000,
            )
            return self._face
        except Exception as exc:
            self._face_error = str(exc)
            logger.error("Face detector could not be initialized: %s", exc)
            return None

    def _detect_faces(self, ctx: FrameContext) -> list[Detection]:
        if not getattr(self.config, "detect_faces", True):
            return []
        detector = self._face_model(ctx.width, ctx.height)
        if detector is None:
            if self._face_error:
                ctx.ocr_errors.append(f"face_detector: {self._face_error}")
            return []
        try:
            _retval, faces = detector.detect(ctx.frame_bgr)
        except Exception as exc:
            logger.error("Face detection failed on frame %s: %s", ctx.frame_number, exc)
            ctx.ocr_errors.append(f"face_detect_frame_{ctx.frame_number}: {exc}")
            return []
        detections: list[Detection] = []
        if faces is None:
            return detections
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            score = float(face[-1]) if len(face) >= 15 else 0.6
            bbox = BBox(x, y, max(1, w), max(1, h)).padded(
                self.config.bbox_padding_px, ctx.width, ctx.height
            )
            detections.append(
                Detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=bbox,
                    category="face",
                    detector="visual.yunet_face",
                    confidence=round(score, 4),
                    review_required=score < self.config.review_confidence_threshold,
                )
            )
        return detections

    def _detect_qr(self, ctx: FrameContext) -> list[Detection]:
        detections: list[Detection] = []
        try:
            ok, decoded, points, _ = self._qr.detectAndDecodeMulti(ctx.frame_bgr)
        except Exception:
            try:
                data, points, _ = self._qr.detectAndDecode(ctx.frame_bgr)
                ok = bool(points is not None)
                decoded = [data] if data else [""]
                points = [points] if points is not None else None
            except Exception as exc:
                logger.error("QR detection failed on frame %s: %s", ctx.frame_number, exc)
                ctx.ocr_errors.append(f"qr_frame_{ctx.frame_number}: {exc}")
                return []
        if not ok or points is None:
            return detections
        for idx, pts in enumerate(np.array(points)):
            xs = pts[:, 0] if pts.ndim == 2 else pts.reshape(-1, 2)[:, 0]
            ys = pts[:, 1] if pts.ndim == 2 else pts.reshape(-1, 2)[:, 1]
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            text = ""
            if decoded is not None and idx < len(decoded):
                text = str(decoded[idx] or "")
            detections.append(
                Detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=BBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1)).padded(
                        self.config.bbox_padding_px, ctx.width, ctx.height
                    ),
                    category="qr_barcode",
                    detector="visual.qr",
                    confidence=0.95 if text else 0.7,
                    text=text or None,
                    review_required=not bool(text),
                )
            )
        return detections

    def _detect_barcodes(self, ctx: FrameContext) -> list[Detection]:
        if self._barcode is None:
            return []
        detections: list[Detection] = []
        try:
            result = self._barcode.detectAndDecode(ctx.frame_bgr)
        except Exception as exc:
            logger.warning("Barcode detection failed on frame %s: %s", ctx.frame_number, exc)
            return []
        decoded = result[1] if len(result) > 1 else None
        points = result[-1] if result else None
        if points is None:
            return detections
        try:
            pts_list = np.array(points)
            if pts_list.ndim == 2:
                pts_list = pts_list[None, ...]
            for idx, pts in enumerate(pts_list):
                pts = np.array(pts).reshape(-1, 2)
                x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                text = ""
                if decoded is not None and idx < len(decoded):
                    text = str(decoded[idx] or "")
                detections.append(
                    Detection(
                        frame_number=ctx.frame_number,
                        timestamp=ctx.timestamp,
                        bbox=BBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1)).padded(
                            self.config.bbox_padding_px, ctx.width, ctx.height
                        ),
                        category="qr_barcode",
                        detector="visual.barcode",
                        confidence=0.9 if text else 0.6,
                        text=text or None,
                        review_required=not bool(text),
                    )
                )
        except Exception as exc:
            logger.warning("Barcode parse failed on frame %s: %s", ctx.frame_number, exc)
        return detections

    def _detect_plate_like(self, ctx: FrameContext) -> list[Detection]:
        """Conservative contour heuristic; always REVIEW_REQUIRED."""
        gray = cv2.cvtColor(ctx.frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blur, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if w < 80 or h < 20 or w > ctx.width * 0.45:
                continue
            ratio = w / float(h)
            if ratio < 1.8 or ratio > 5.5:
                continue
            if y < ctx.height * 0.35:
                continue
            roi = gray[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            if cv2.mean(roi)[0] < 80:
                continue
            detections.append(
                Detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=BBox(x, y, w, h).padded(self.config.bbox_padding_px, ctx.width, ctx.height),
                    category="license_plate",
                    detector="visual.plate_contour",
                    confidence=0.4,
                    review_required=True,
                )
            )
            if len(detections) >= 8:
                break
        return detections
