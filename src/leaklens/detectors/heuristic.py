from __future__ import annotations

import re

import cv2
import numpy as np

from leaklens.classify import make_detection
from leaklens.config import AppConfig
from leaklens.detectors.base import FrameContext
from leaklens.detectors.regex import NAME_UI_WORDS, _artish_ratio, _valid_user_host, is_ascii_art
from leaklens.models import BBox, Detection, OcrItem
from leaklens.util import slice_bbox, union_boxes

IDENTITY_HINTS = re.compile(
    r"\b(?:password|passwd|token|secret|api[_-]?key|authorization|cookie|"
    r"session|ssid|hostname|username|user@|root@|signed in|welcome back|"
    r"verification code|one-time|otp|account|ipv4|ipv6|mac address)\b",
    re.I,
)
DEMO_RESULTS_RE = re.compile(
    r"attacker captured|captured at|check the url|wasn't instagram|"
    r"look for missing padlock|password for demos|use a password manager",
    re.I,
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

PROMPT_RE = re.compile(
    r"(?:\(([A-Za-z][A-Za-z0-9._-]{0,31})@([A-Za-z0-9][A-Za-z0-9._-]{1,63})\)|"
    r"([A-Za-z][A-Za-z0-9._-]{0,31})@([A-Za-z0-9][A-Za-z0-9._-]{1,63})\s*[:#~$])"
)
MERGED_PROMPT_RE = re.compile(
    r"\(?([A-Za-z][a-z0-9._-]{1,20})([A-Z][A-Z0-9._-]{2,31})\)\s*-\s*\["
)
PROMPT_LEFT_RE = re.compile(r"\(?([A-Za-z][A-Za-z0-9._-]{1,20})$")
PROMPT_RIGHT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9._-]{1,20})\)\s*-\s*\[")


class HeuristicDetector:
    name = "heuristic"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def detect(self, ctx: FrameContext) -> list[Detection]:
        detections: list[Detection] = []
        detections.extend(self._title_bar(ctx))
        detections.extend(self._mail_tabs(ctx))
        detections.extend(self._prompts(ctx))
        detections.extend(self._notifications(ctx))
        detections.extend(self._autofill(ctx))
        detections.extend(self._email_stack(ctx))
        detections.extend(self._identity_popup(ctx))
        return detections

    def _title_bar(self, ctx: FrameContext) -> list[Detection]:
        top = [item for item in ctx.ocr_items if item.bbox.y <= int(ctx.height * 0.12)]
        hits: list[Detection] = []
        for item in top:
            if not item.text or is_ascii_art(item.text) or _artish_ratio(item.text) >= 0.7:
                continue
            match = PROMPT_RE.search(item.text)
            if not match:
                continue
            user = match.group(1) or match.group(3)
            host = match.group(2) or match.group(4)
            text = f"{user}@{host}"
            if not _valid_user_host(text):
                continue
            bbox = _secret_bbox(item, text)
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=bbox.padded(self.config.bbox_padding_px, ctx.width, ctx.height),
                    category="hostname",
                    detector="heuristic.title_bar",
                    confidence=0.84,
                    text=text,
                    classification="SENSITIVE",
                )
            )
        return hits

    def _mail_tabs(self, ctx: FrameContext) -> list[Detection]:
        """Gmail/Inbox tab titles often OCR as 'Inbox (N) - username' without @domain."""
        top = int(ctx.height * 0.10)
        tab_items = [item for item in ctx.ocr_items if item.text and item.bbox.y <= top]
        if not tab_items:
            return []
        hits: list[Detection] = []
        pattern = re.compile(
            r"(?:inbox|gmail|outlook|yahoo\s+mail|mail)"
            r"(?:\s*\(\s*[\d,]+\s*\))?\s*[-–—]\s*([A-Za-z][A-Za-z0-9._]{2,40})",
            re.I,
        )
        for item in tab_items:
            cleaned = item.text.replace("MInbox", "Inbox").replace("M Inbox", "Inbox")
            match = pattern.search(cleaned)
            token = match.group(1) if match else re.sub(r"[^A-Za-z0-9._-]+", "", item.text)
            if re.search(r"https?://|sslip\.io|www\.", item.text, re.I) or len(token) > 40:
                continue
            if not match and not re.fullmatch(r"[A-Za-z][A-Za-z0-9._]{3,30}\d[A-Za-z0-9._]*", token):
                continue
            bbox = _secret_bbox(item, token)
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=bbox.padded(max(14, self.config.bbox_padding_px + 8), ctx.width, ctx.height),
                    category="username_handle",
                    detector="heuristic.mail_tab",
                    confidence=0.9,
                    text=token,
                    classification="SENSITIVE",
                )
            )
        return hits

    def _prompts(self, ctx: FrameContext) -> list[Detection]:
        hits: list[Detection] = []
        items = [item for item in ctx.ocr_items if item.text and not is_ascii_art(item.text)]
        for item in items:
            user = host = None
            merged = MERGED_PROMPT_RE.search(item.text)
            prompt = PROMPT_RE.search(item.text)
            if merged:
                user, host = merged.group(1), merged.group(2)
            elif prompt:
                user = prompt.group(1) or prompt.group(3)
                host = prompt.group(2) or prompt.group(4)
            if not user or not host:
                continue
            text = f"{user}@{host}"
            if not _valid_user_host(text):
                continue
            bbox = _secret_bbox(item, text)
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=bbox.padded(self.config.bbox_padding_px, ctx.width, ctx.height),
                    category="hostname",
                    detector="heuristic.shell_prompt",
                    confidence=0.91,
                    text=text,
                    classification="SENSITIVE",
                )
            )
        for left, right in zip(items, items[1:]):
            if abs((left.bbox.y + left.bbox.h / 2) - (right.bbox.y + right.bbox.h / 2)) > 14:
                continue
            m1 = PROMPT_LEFT_RE.search(left.text.strip())
            m2 = PROMPT_RIGHT_RE.search(right.text.strip())
            if not m1 or not m2:
                continue
            text = f"{m1.group(1)}@{m2.group(1)}"
            if not _valid_user_host(text):
                continue
            merged_box = union_boxes([left.bbox, right.bbox])
            if merged_box is None:
                continue
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=merged_box.padded(self.config.bbox_padding_px, ctx.width, ctx.height),
                    category="hostname",
                    detector="heuristic.shell_prompt_split",
                    confidence=0.86,
                    text=text,
                    classification="REVIEW_REQUIRED",
                    extra={"reason": "ocr_dropped_at_sign"},
                )
            )
        return hits

    def _notifications(self, ctx: FrameContext) -> list[Detection]:
        """Flag bright top-of-screen banners that also contain identity-like OCR."""
        frame = ctx.frame_bgr
        if frame is None:
            return []
        top_h = max(40, int(ctx.height * 0.18))
        region = frame[0:top_h, :]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        sat = hsv[:, :, 1]
        mask = ((value > 180) & (sat > 40)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hits: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < ctx.width * 0.18 or h < 24 or h > top_h * 0.9:
                continue
            box = BBox(x, y, w, h)
            overlapping = [
                item
                for item in ctx.ocr_items
                if item.bbox.iou(box) > 0.05 or _contains(box, item.bbox)
            ]
            text = " ".join(item.text for item in overlapping if item.text)
            if not text or not IDENTITY_HINTS.search(text):
                continue
            if not re.search(
                r"(\d{1,3}\.){3}\d{1,3}|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|eyJ[A-Za-z0-9_-]{8,}",
                text,
            ):
                continue
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=box.padded(self.config.bbox_padding_px, ctx.width, ctx.height),
                    category="notification_popup",
                    detector="heuristic.notification",
                    confidence=0.7,
                    text=text[:240],
                    classification="REVIEW_REQUIRED",
                )
            )
        return hits

    def _autofill(self, ctx: FrameContext) -> list[Detection]:
        """Stacked similar-width OCR boxes often indicate browser autofill."""
        items = [i for i in ctx.ocr_items if i.text and not is_ascii_art(i.text)]
        items = [i for i in items if i.bbox.y > int(ctx.height * 0.15)]
        items = sorted(items, key=lambda i: i.bbox.y)
        if _is_demo_results_page(items):
            return []
        hits: list[Detection] = []
        for i, first in enumerate(items):
            cluster = [first]
            for other in items[i + 1 : i + 6]:
                if abs(other.bbox.x - first.bbox.x) > 40:
                    continue
                if abs(other.bbox.w - first.bbox.w) > max(40, first.bbox.w * 0.45):
                    continue
                gap = other.bbox.y - (cluster[-1].bbox.y + cluster[-1].bbox.h)
                if gap < -5 or gap > 28:
                    continue
                cluster.append(other)
            if len(cluster) < 3:
                continue
            text = " | ".join(item.text for item in cluster)
            if not EMAIL_RE.search(text):
                continue
            if DEMO_RESULTS_RE.search(text):
                continue
            bbox = union_boxes([item.bbox for item in cluster])
            if bbox is None:
                continue
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=bbox.padded(self.config.bbox_padding_px, ctx.width, ctx.height),
                    category="autofill_suggestion",
                    detector="heuristic.autofill",
                    confidence=0.58,
                    text=text[:240],
                    classification="REVIEW_REQUIRED",
                )
            )
        return hits

    def _email_stack(self, ctx: FrameContext) -> list[Detection]:
        """Browser autofill lists are often 2+ emails, not a 3-box password cluster."""
        email_re = re.compile(
            r"[A-Za-z0-9._%+\-]{2,}@(?:[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|"
            r"(?:gmail|proton(?:mail)?|outlook|hotmail|yahoo|icloud)\.?[A-Za-z]{0,10})",
            re.I,
        )
        emails = [
            item
            for item in ctx.ocr_items
            if item.text and email_re.search(item.text) and not is_ascii_art(item.text)
        ]
        if _is_demo_results_page(ctx.ocr_items):
            return []
        emails.sort(key=lambda i: i.bbox.y)
        hits: list[Detection] = []
        used: set[int] = set()
        for i, first in enumerate(emails):
            if i in used:
                continue
            cluster = [first]
            used.add(i)
            for j, other in enumerate(emails[i + 1 :], start=i + 1):
                if j in used:
                    continue
                if abs(other.bbox.x - first.bbox.x) > 80:
                    continue
                gap = other.bbox.y - (cluster[-1].bbox.y + cluster[-1].bbox.h)
                if gap < -8 or gap > 80:
                    continue
                cluster.append(other)
                used.add(j)
            if len(cluster) < 2:
                continue
            bbox = union_boxes([item.bbox for item in cluster])
            if bbox is None:
                continue
            text = " | ".join(item.text for item in cluster)
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=bbox.padded(max(18, self.config.bbox_padding_px + 12), ctx.width, ctx.height),
                    category="autofill_suggestion",
                    detector="heuristic.email_stack",
                    confidence=0.88,
                    text=text[:240],
                    classification="SENSITIVE",
                )
            )
        return hits

    def _identity_popup(self, ctx: FrameContext) -> list[Detection]:
        """Redact the Chrome 'Saved info' panel, including username-only rows.

        Require a Saved-info header or a real email. Do not treat OBS chrome,
        the Instagram logo, or the phishing demo results panel as a popup.
        """
        items = [i for i in ctx.ocr_items if i.text and not is_ascii_art(i.text)]
        if not items:
            return []
        chrome_y = int(ctx.height * 0.15)
        page_items = [i for i in items if i.bbox.y > chrome_y]
        if _is_demo_results_page(items):
            return []
        headers = [i for i in page_items if _looks_like_saved_info(i.text)]
        emails = [i for i in page_items if _is_popup_email(i.text)]
        if not headers and not emails:
            return []
        hits: list[Detection] = []
        used: set[int] = set()
        anchors = headers + emails
        for anchor in anchors:
            if id(anchor) in used:
                continue
            column = [anchor]
            used.add(id(anchor))
            top = anchor.bbox.y - 80
            bottom = anchor.bbox.y + anchor.bbox.h + 260
            for other in page_items:
                if id(other) in used:
                    continue
                center_delta = abs(
                    (other.bbox.x + other.bbox.w / 2) - (anchor.bbox.x + anchor.bbox.w / 2)
                )
                if abs(other.bbox.x - anchor.bbox.x) > 90 and center_delta > 110:
                    continue
                if other.bbox.y < top or other.bbox.y > bottom:
                    continue
                if (
                    _looks_like_saved_info(other.text)
                    or _is_popup_email(other.text)
                    or _is_popup_identity(other.text)
                    or _is_popup_chrome(other.text)
                ):
                    column.append(other)
                    used.add(id(other))
                    bottom = max(bottom, other.bbox.y + other.bbox.h + 40)
            has_header = any(_looks_like_saved_info(it.text) for it in column)
            has_email = any(_is_popup_email(it.text) for it in column)
            if not has_header and not has_email:
                continue
            if DEMO_RESULTS_RE.search(" ".join(it.text for it in column)):
                continue
            bbox = union_boxes([it.bbox for it in column])
            if bbox is None:
                continue
            # Cover the associated login input sitting just above Saved info.
            lift = 80
            y = max(chrome_y, bbox.y - lift)
            bbox = BBox(
                max(0, bbox.x - 52),
                y,
                bbox.w + 76,
                bbox.y + bbox.h + 36 - y,
            )
            bbox = BBox(bbox.x, bbox.y, max(bbox.w, 260), max(bbox.h, 90))
            if bbox.y < chrome_y:
                continue
            if bbox.w * bbox.h > ctx.width * ctx.height * 0.18:
                continue
            bbox = expand_to_dark_panel(ctx.frame_bgr, bbox, ctx.width, ctx.height)
            hits.append(
                make_detection(
                    frame_number=ctx.frame_number,
                    timestamp=ctx.timestamp,
                    bbox=bbox.padded(8, ctx.width, ctx.height),
                    category="autofill_suggestion",
                    detector="heuristic.identity_popup",
                    confidence=0.94,
                    text=" | ".join(it.text for it in column)[:240],
                    classification="SENSITIVE",
                )
            )
        return hits


def _looks_like_saved_info(text: str) -> bool:
    compact = re.sub(r"[^a-z]", "", (text or "").lower())
    return compact.endswith("info") and compact.startswith("s") and 6 <= len(compact) <= 14


def _is_demo_results_page(items: list[OcrItem]) -> bool:
    blob = " ".join(item.text for item in items if item.text)
    return bool(DEMO_RESULTS_RE.search(blob))


def _is_popup_chrome(text: str) -> bool:
    return bool(re.search(r"manage\s+personal|password manager", text or "", re.I))


_POPUP_SKIP = {
    "password", "passw", "instagram", "facebook", "email", "saved", "info",
    "login", "or", "sign", "signup", "forgot", "username", "user", "name",
    "scene", "scenes", "mixer", "audio", "studio", "obs", "untitled",
    "profile", "docks", "controls", "fade", "duration", "options", "global",
    "settings", "tools", "help", "file", "edit", "view", "desktop", "capture",
    "streaming", "recording", "virtual", "collection", "transitions",
    "textnow", "dashboard", "hetzner", "telnyx", "mic", "aux", "properties",
    "filters", "display", "mode", "start", "stop", "manage", "personal",
}


def _is_popup_email(text: str) -> bool:
    raw = (text or "").strip()
    if EMAIL_RE.search(raw):
        return True
    return bool(
        re.search(
            r"[A-Za-z0-9._%+\-]{2,64}@(?:gmail|googlemail|proton(?:mail)?|outlook|"
            r"hotmail|yahoo|icloud|live|msn|aol|me)\.?",
            raw,
            re.I,
        )
    )


def _is_popup_identity(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or raw.lower() in _POPUP_SKIP:
        return False
    if "/" in raw or "|" in raw:
        return False
    if " " in raw and "@" not in raw and not re.search(r"gmail\.|proton\.|outlook\.", raw, re.I):
        return False
    if _is_popup_email(raw):
        return True
    if "@" in raw:
        return False
    token = re.sub(r"[^A-Za-z0-9._-]+", "", raw)
    lowered = token.lower()
    if lowered in _POPUP_SKIP or lowered in NAME_UI_WORDS:
        return False
    if token.count("-") >= 2:
        return False
    if "-" in token and re.search(r"\d", token):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{2,40}", token))


def expand_to_dark_panel(frame, box: BBox, frame_w: int, frame_h: int) -> BBox:
    """Grow a box to the dark rounded Chrome autofill panel around it."""
    if frame is None or box.w < 2 or box.h < 2:
        return box
    chrome_y = int(frame_h * 0.14)
    if box.y < chrome_y:
        return box
    try:
        image = np.ascontiguousarray(frame)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box.as_xyxy()
        cx = min(width - 1, max(0, (x1 + x2) // 2))
        cy = min(height - 1, max(0, (y1 + y2) // 2))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = np.ascontiguousarray(hsv[:, :, 2])
        if int(value[cy, cx]) > 100:
            return box
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(
            value,
            mask,
            (int(cx), int(cy)),
            255,
            loDiff=22,
            upDiff=22,
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
        )
        filled = mask[1 : height + 1, 1 : width + 1]
        ys, xs = np.where(filled > 0)
        if xs.size < 800:
            return box
        nx1, nx2 = int(np.min(xs)), int(np.max(xs)) + 1
        ny1, ny2 = int(np.min(ys)), int(np.max(ys)) + 1
        if ny1 < chrome_y:
            return box
        if ny1 < box.y - 40:
            return box
        if (nx2 - nx1) > frame_w * 0.42 or (ny2 - ny1) > frame_h * 0.38:
            return box
        if (nx2 - nx1) < box.w * 0.75:
            return box
        return BBox(nx1, ny1, nx2 - nx1, ny2 - ny1)
    except Exception:
        return box


def _secret_bbox(item: OcrItem, secret: str) -> BBox:
    haystack = item.text or ""
    idx = haystack.lower().find(secret.lower())
    if idx < 0:
        return item.bbox
    return slice_bbox(item.bbox, len(haystack), idx, idx + len(secret))


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x + inner.w <= outer.x + outer.w
        and inner.y + inner.h <= outer.y + outer.h
    )
