from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from leaklens.classify import make_detection
from leaklens.config import AppConfig
from leaklens.detectors.base import FrameContext
from leaklens.models import BBox, Detection, OcrItem
from leaklens.util import (
    boxes_for_span,
    build_char_map,
    collapse_overlapping_ocr,
    shannon_entropy,
    slice_bbox,
    union_boxes,
)

logger = logging.getLogger(__name__)

NAME_STOPWORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "fri aug", "mon jan", "expanded security",
    "welcome", "ubuntu", "linux", "system", "usage", "users", "learn", "apps",
    "last", "security", "information", "kernel", "updates", "packages",
    "new york", "los angeles", "san francisco", "united states", "north america",
    "google chrome", "visual studio", "command prompt", "windows terminal",
    "not found", "permission denied", "access denied", "file explorer",
    "click here", "read more", "see more", "dark mode", "light mode",
    "error code", "status code", "home page", "sign in", "log in",
    "app store", "google play", "social engineering", "video tip",
    "studio mode", "display capture", "audio mixer", "start streaming",
    "desktop audio", "start virtual", "stop recording", "tools help",
    "scene collection", "scene transitions", "file edit", "privacy help",
    "threads contact", "uploading terms", "security demo", "you just",
    "forgot password", "get the", "download on",
}

NAME_UI_WORDS = {
    "app", "store", "google", "play", "download", "social", "engineering",
    "video", "tip", "studio", "mode", "display", "capture", "audio", "mixer",
    "start", "stop", "streaming", "desktop", "file", "edit", "view", "tools",
    "help", "scene", "scenes", "collection", "privacy", "terms", "threads",
    "contact", "uploading", "security", "demo", "virtual", "recording",
    "password", "forgot", "manage", "personal", "info", "profile", "docks",
    "transitions", "untitled", "instagram", "youtube", "chrome", "firefox",
    "windows", "terminal", "command", "prompt", "dashboard", "settings",
    "meta", "verified", "customer", "telnyx", "hetzner", "textnow", "fade",
    "duration", "options", "global", "controls", "save", "not", "properties",
    "filters", "mic", "aux", "obs", "docks", "passw", "edt", "wew", "fle",
}

KNOWN_MAIL_DOMAINS = (
    "gmail", "googlemail", "proton", "protonmail", "outlook", "hotmail",
    "yahoo", "icloud", "live", "msn", "aol", "me", "gmx", "mail",
)

COMMON_BINARIES = {
    "ssh", "scp", "rsync", "curl", "wget", "sudo", "bash", "zsh", "python",
    "node", "docker", "kubectl", "git", "npm", "pip", "apt", "yum", "brew",
    "systemctl", "journalctl", "nmap", "ping", "tracepath", "traceroute",
}

STREET_SUFFIX = (
    r"Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Way|Place|Pl|Circle|Cir|Parkway|Pkwy|Highway|Hwy|Terrace|Ter"
)


@dataclass(frozen=True)
class PatternRule:
    name: str
    category: str
    pattern: re.Pattern[str]
    confidence: float
    review_required: bool = False
    validator: Callable[[str], bool] | None = None
    skip_ascii_art: bool = True
    capture_group: int | None = None
    slice_match: bool = True


def _compile(pattern: str, flags: int = re.I) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def _is_private_ip(text: str) -> bool | None:
    try:
        ip = ipaddress.ip_address(text.strip().rstrip(".,;:)"))
    except ValueError:
        return None
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)


def _valid_ipv4(text: str) -> bool:
    candidate = text.strip().rstrip(".,;:)")
    try:
        ipaddress.IPv4Address(candidate)
    except ValueError:
        return False
    return candidate not in {"0.0.0.0", "255.255.255.255"}


def _valid_ipv6(text: str) -> bool:
    candidate = text.strip().rstrip(".,;:)%")
    if candidate.count(":") < 2:
        return False
    try:
        ipaddress.IPv6Address(candidate.split("%")[0])
        return True
    except ValueError:
        return False


def _valid_mac(text: str) -> bool:
    hex_parts = re.split(r"[:-]", text.strip())
    if len(hex_parts) != 6:
        return False
    joined = "".join(hex_parts).lower()
    return joined not in {"000000000000", "ffffffffffff"}


def _valid_email(text: str) -> bool:
    return "@" in text and "." in text.split("@")[-1]


def _valid_plate(text: str) -> bool:
    raw = (text or "").strip()
    if not re.search(r"[A-Z]{1,3}[ -]\d", raw):
        return False
    compact = re.sub(r"[\s-]", "", raw)
    if "000" in compact:
        return False
    if len(compact) < 5 or len(compact) > 8:
        return False
    return any(ch.isalpha() for ch in compact) and any(ch.isdigit() for ch in compact)


def _valid_name(text: str) -> bool:
    lowered = " ".join(text.split()).lower()
    if lowered in NAME_STOPWORDS:
        return False
    parts = text.split()
    if len(parts) != 2:
        return False
    if any(part.lower() in NAME_STOPWORDS or part.lower() in NAME_UI_WORDS for part in parts):
        return False
    if parts[0].lower() == parts[1].lower():
        return False
    return all(p[0].isupper() and p[1:].islower() and p.isalpha() for p in parts)


def _valid_tab_identity(text: str) -> bool:
    token = re.sub(r"[^A-Za-z0-9._-]+", "", text or "")
    if len(token) < 3:
        return False
    return token.lower() not in {"inbox", "gmail", "mail", "outlook", "instagram", "chrome"}


def _valid_truncated_email(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if "@" not in compact:
        return False
    local, _, domain = compact.partition("@")
    if len(local) < 2:
        return False
    brand = domain.lower().split(".", 1)[0]
    return brand in KNOWN_MAIL_DOMAINS


def _artish_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    art = sum(ch in "@#%*+=.:;~^|\\/[]{}()<>0OQCEAMWXeocq" for ch in compact)
    return art / len(compact)


def is_ascii_art(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 16:
        alnum = sum(ch.isalnum() or ch.isspace() for ch in text)
        symbols = len(text) - alnum
        if symbols / len(text) >= 0.45:
            return True
    if len(compact) >= 12 and _artish_ratio(compact) >= 0.75:
        return True
    return False


def _not_binary(text: str) -> bool:
    user = text.split("@")[0].lower()
    return user not in COMMON_BINARIES


def _valid_user_host(text: str) -> bool:
    if text.count("@") != 1:
        return False
    user, host = text.split("@", 1)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,31}", user):
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,63}", host):
        return False
    if not re.search(r"[A-Za-z]", host):
        return False
    if is_ascii_art(text) or _artish_ratio(text) >= 0.7:
        return False
    return _not_binary(text)


def _valid_handle(text: str) -> bool:
    token = text.lstrip("@")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._]{2,30}", token):
        return False
    if not re.search(r"[aeiouyAEIOUY]", token):
        return False
    return _artish_ratio(token) < 0.7


def _valid_password_value(text: str) -> bool:
    value = text.strip().strip("\"'`")
    if ":" in value and not value.lower().startswith("password"):
        value = re.split(r"[:=]", value, maxsplit=1)[-1].strip().strip("\"'`")
    lowered = value.lower().rstrip(":=")
    if lowered in {"password", "passwd", "pwd", "passphrase", "welcome", "yes", "no", "true", "false"}:
        return False
    if re.fullmatch(r"[*•·xX]+", value):
        return False
    if value in {"|", ">", "#", "$", "%"}:
        return False
    if "'" in value or "@" in value or _valid_ipv4(value):
        return False
    return len(value) >= 3


def _valid_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if len(digits) < 10 or len(digits) > 15:
        return False
    if len(set(digits)) <= 1 or digits.startswith("000"):
        return False
    return True


RULES: list[PatternRule] = [
    PatternRule(
        "jwt",
        "jwt",
        _compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]{8,})?\b", 0),
        0.97,
    ),
    PatternRule(
        "pem_private_key",
        "ssh_key",
        _compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----"),
        0.99,
    ),
    PatternRule(
        "ssh_key_blob",
        "ssh_key",
        _compile(r"\b(?:ssh-(?:rsa|ed25519|dss)|ecdsa-sha2-nistp(?:256|384|521))\s+AAAA[A-Za-z0-9+/=]{20,}"),
        0.96,
    ),
    PatternRule(
        "aws_access_key",
        "cloud_credential",
        _compile(r"\b(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b", 0),
        0.98,
    ),
    PatternRule(
        "github_token",
        "access_token",
        _compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|(?:github_pat_[A-Za-z0-9_]{20,})", 0),
        0.98,
    ),
    PatternRule(
        "slack_token",
        "access_token",
        _compile(r"\bxox[baprs]-[\w-]{10,}\b", 0),
        0.97,
    ),
    PatternRule(
        "stripe_key",
        "api_key",
        _compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b", 0),
        0.97,
    ),
    PatternRule(
        "google_api_key",
        "api_key",
        _compile(r"\bAIza[0-9A-Za-z\-_]{35}\b", 0),
        0.96,
    ),
    PatternRule(
        "openai_key",
        "api_key",
        _compile(r"\bsk-[A-Za-z0-9]{20,}\b", 0),
        0.9,
        review_required=True,
    ),
    PatternRule(
        "sendgrid_key",
        "api_key",
        _compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b", 0),
        0.95,
    ),
    PatternRule(
        "generic_api_assignment",
        "api_key",
        _compile(
            r"\b(?:api[_-]?key|access[_-]?key|secret[_-]?key|client[_-]?secret|private[_-]?key)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9/+=._-]{12,}"
        ),
        0.9,
    ),
    PatternRule(
        "bearer_token",
        "authorization_header",
        _compile(r"\b(?:Authorization\s*[:=]\s*)?(?:Bearer|Basic)\s+[A-Za-z0-9._\-+/=]{12,}"),
        0.94,
    ),
    PatternRule(
        "cookie_header",
        "cookie_session",
        _compile(
            r"\b(?:Set-Cookie|Cookie)\s*[:=]\s*[^\s]{8,}|"
            r"\b(?:session(?:id)?|sid|jsessionid|auth[_-]?token|csrf(?:token)?)\s*[:=]\s*[A-Za-z0-9._-]{8,}"
        ),
        0.9,
    ),
    PatternRule(
        "db_connection",
        "database_connection",
        _compile(
            r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|"
            r"amqp|amqps|mssql|sqlserver|jdbc:(?:mysql|postgresql)|odbc)"
            r":\/\/[^\s'\"<>]{8,}"
        ),
        0.96,
    ),
    PatternRule(
        "password_assignment",
        "password",
        _compile(r"\b(?:password|passwd|pwd|passphrase)\s*[:=]\s*(\S{3,})"),
        0.88,
        validator=_valid_password_value,
        capture_group=1,
    ),
    PatternRule(
        "windows_user_path",
        "windows_path",
        _compile(r"\b[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\s\\/:*?\"<>|]{1,64}"),
        0.93,
    ),
    PatternRule(
        "unix_home_path",
        "unix_home_path",
        _compile(r"(?:/home|/Users)/[A-Za-z0-9._-]+"),
        0.92,
    ),
    PatternRule(
        "truncated_email",
        "email",
        _compile(
            r"\b[A-Za-z0-9._%+\-]{2,64}@(?:gmail|googlemail|proton(?:mail)?|outlook|"
            r"hotmail|yahoo|icloud|live|msn|aol|me)\.?[A-Za-z]{0,10}\b"
        ),
        0.9,
        validator=_valid_truncated_email,
    ),
    PatternRule(
        "email",
        "email",
        _compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b"),
        0.93,
        validator=_valid_email,
    ),
    PatternRule(
        "mail_tab_identity",
        "username_handle",
        _compile(
            r"(?:inbox|gmail|outlook|yahoo\s+mail|mail)"
            r"(?:\s*\(\s*[\d,]+\s*\))?\s*[-–—]\s*([A-Za-z][A-Za-z0-9._]{2,40})"
        ),
        0.92,
        capture_group=1,
        validator=_valid_tab_identity,
    ),
    PatternRule(
        "user_at_host",
        "hostname",
        _compile(r"\b[A-Za-z0-9._\-]{1,32}@[A-Za-z0-9._\-]{1,64}\b"),
        0.9,
        validator=_valid_user_host,
    ),
    PatternRule(
        "ipv4",
        "ipv4",
        _compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b"),
        0.9,
        validator=_valid_ipv4,
    ),
    PatternRule(
        "ip_wildcard_dns",
        "private_url",
        _compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})[.-]){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})"
            r"\.(?:sslip|nip|xip)\.io\b"
        ),
        0.96,
        slice_match=False,
    ),
    PatternRule(
        "ipv6",
        "ipv6",
        _compile(
            r"\b(?:(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}|::1|fe80:(?:[0-9a-f:]*)+|::(?:ffff:)?(?:\d{1,3}\.){3}\d{1,3})\b"
        ),
        0.88,
        validator=_valid_ipv6,
    ),
    PatternRule(
        "mac_address",
        "mac_address",
        _compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b", 0),
        0.9,
        validator=_valid_mac,
    ),
    PatternRule(
        "gps",
        "gps_coordinates",
        _compile(
            r"\b-?(?:[1-8]?\d(?:\.\d{3,})?|90(?:\.0{3,})?)\s*[,;]\s*-?(?:1[0-7]\d(?:\.\d{3,})?|[1-9]?\d(?:\.\d{3,})?|180(?:\.0{3,})?)\b"
        ),
        0.8,
        review_required=True,
    ),
    PatternRule(
        "phone",
        "phone_number",
        _compile(
            r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
        ),
        0.72,
        review_required=True,
        validator=_valid_phone,
    ),
    PatternRule(
        "street_address",
        "street_address",
        _compile(rf"\b\d{{1,6}}\s+[A-Z][A-Za-z0-9.'\-]+\s+(?:{STREET_SUFFIX})\.?\b"),
        0.8,
        review_required=True,
    ),
    PatternRule(
        "handle",
        "username_handle",
        _compile(r"(?<![A-Za-z0-9._])@[A-Za-z][A-Za-z0-9._]{2,30}\b", 0),
        0.55,
        review_required=True,
        validator=_valid_handle,
    ),
    PatternRule(
        "credential_url",
        "private_url",
        _compile(r"https?://[^\s/:]+:[^\s/@]+@[^\s'\"<>]+"),
        0.96,
    ),
    PatternRule(
        "private_url",
        "private_url",
        _compile(
            r"https?://(?:localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
            r"[A-Za-z0-9._\-]+\.(?:internal|local|lan|corp|home|intranet))(?::\d+)?[^\s'\"<>]*"
        ),
        0.93,
    ),
    PatternRule(
        "internal_hostname",
        "hostname",
        _compile(r"\b[A-Za-z0-9][A-Za-z0-9\-.]{1,60}\.(?:local|lan|internal|corp|home|intranet)\b"),
        0.9,
    ),
    PatternRule(
        "wifi_ssid",
        "wifi_ssid",
        _compile(r"\b(?:SSID|Wi-?Fi(?:\s+name)?|Network(?:\s+name)?)\s*[:=#\-]?\s*['\"]?([^\s'\"]{2,32})"),
        0.85,
        review_required=True,
    ),
    PatternRule(
        "license_plate_text",
        "license_plate",
        _compile(r"\b[A-Z]{1,3}[ -]?\d{1,4}[ -]?[A-Z]{1,3}\b", 0),
        0.45,
        review_required=True,
        validator=_valid_plate,
    ),
    PatternRule(
        "full_name",
        "full_name",
        _compile(r"\b[A-Z][a-z]{2,20}\s+[A-Z][a-z]{2,20}\b", 0),
        0.5,
        review_required=True,
        validator=_valid_name,
    ),
]


class RegexDetector:
    name = "regex"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def detect(self, ctx: FrameContext) -> list[Detection]:
        detections: list[Detection] = []
        items = collapse_overlapping_ocr(
            [item for item in ctx.ocr_items if item.text and item.bbox.h <= 100]
        )
        if not items:
            return detections

        for row in _group_rows(items):
            detections.extend(self._apply_rules(ctx, row))
        detections.extend(self._high_entropy(ctx, items))
        return detections

    def _apply_rules(self, ctx: FrameContext, items: list[OcrItem]) -> list[Detection]:
        detections: list[Detection] = []
        blob, spans = build_char_map(items)
        for rule in RULES:
            for match in rule.pattern.finditer(blob):
                span_start, span_end = match.start(), match.end()
                raw = match.group(0)
                if rule.capture_group:
                    captured = match.group(rule.capture_group)
                    if not captured:
                        continue
                    span_start, span_end = match.start(rule.capture_group), match.end(rule.capture_group)
                    raw = captured
                if rule.validator and not rule.validator(raw):
                    continue
                matched_items = [
                    span[2]
                    for span in spans
                    if span[1] > span_start and span[0] < span_end
                ]
                if rule.skip_ascii_art and matched_items and all(is_ascii_art(i.text) for i in matched_items):
                    continue
                cover_url = (
                    rule.category in {"ipv4", "ipv6"} or rule.name == "ip_wildcard_dns"
                ) and _should_cover_url_host(blob, span_start, span_end)
                slice_glyphs = (
                    rule.slice_match
                    and not self.config.expand_match_to_ocr_line
                    and not cover_url
                )
                boxes = boxes_for_span(
                    spans,
                    span_start,
                    span_end,
                    slice_glyphs=slice_glyphs,
                )
                if cover_url:
                    host_boxes = _ip_host_boxes(items, ctx.width, ctx.height)
                    if host_boxes:
                        boxes = host_boxes
                bbox = union_boxes(boxes)
                if bbox is None:
                    continue
                pad = self.config.bbox_padding_px
                if rule.category in {"email", "username_handle"} or not slice_glyphs:
                    pad = max(pad + 10, 14)
                if cover_url:
                    pad = max(pad, 10)
                category = _refine_category(rule.category, raw)
                confidence = min(1.0, rule.confidence * _ocr_confidence(matched_items))
                classification = "REVIEW_REQUIRED" if (
                    rule.review_required or confidence < self.config.review_confidence_threshold
                ) else "SENSITIVE"
                detections.append(
                    make_detection(
                        frame_number=ctx.frame_number,
                        timestamp=ctx.timestamp,
                        bbox=bbox.padded(pad, ctx.width, ctx.height),
                        category=category,
                        detector=f"regex.{rule.name}",
                        confidence=confidence,
                        text=raw,
                        classification=classification,
                    )
                )
        return detections

    def _high_entropy(self, ctx: FrameContext, items: list[OcrItem]) -> list[Detection]:
        found: list[Detection] = []
        token_re = re.compile(r"[A-Za-z0-9/+=._\-]{20,}")
        for item in items:
            if is_ascii_art(item.text):
                continue
            for token in token_re.findall(item.text):
                if token.startswith("-----"):
                    continue
                if _looks_like_public_url(token):
                    continue
                entropy = shannon_entropy(token)
                if entropy < self.config.high_entropy_threshold:
                    continue
                if token.isalpha() or token.isdigit():
                    continue
                idx = item.text.find(token)
                bbox = item.bbox
                if idx >= 0:
                    bbox = slice_bbox(item.bbox, len(item.text), idx, idx + len(token))
                found.append(
                    make_detection(
                        frame_number=ctx.frame_number,
                        timestamp=ctx.timestamp,
                        bbox=bbox.padded(self.config.bbox_padding_px, ctx.width, ctx.height),
                        category="other_sensitive",
                        detector="regex.high_entropy",
                        confidence=min(0.8, 0.4 + entropy / 8),
                        text=token,
                        classification="REVIEW_REQUIRED",
                        extra={"entropy": round(entropy, 3)},
                    )
                )
        return found


def _ocr_confidence(items: list[OcrItem]) -> float:
    if not items:
        return 0.7
    return max(0.35, min(1.0, sum(i.confidence for i in items) / len(items)))


def _refine_category(category: str, text: str) -> str:
    if category != "ipv4":
        return category
    private = _is_private_ip(text)
    if private is True:
        return "ipv4_private"
    if private is False:
        return "ipv4_public"
    return "ipv4"


def _ip_embedded_in_hostname(blob: str, start: int, end: int) -> bool:
    """True when the IPv4 sits inside a host like 203.0.113.10.sslip.io."""
    before = blob[start - 1] if start > 0 else ""
    after = blob[end] if end < len(blob) else ""
    if before == "." or after == ".":
        return True
    hostish = blob[max(0, start - 32) : min(len(blob), end + 32)].lower()
    return any(token in hostish for token in ("sslip", "nip.io", "xip.io", ".com."))


def _should_cover_url_host(blob: str, start: int, end: int) -> bool:
    lowered = blob.lower()
    if any(token in lowered for token in ("sslip", "nip.io", "xip.io")):
        return True
    return _ip_embedded_in_hostname(blob, start, end)


_IP_HOST_RE = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})"
    r"(?:\.(?:sslip|nip|xip)\.io)?",
    re.I,
)


def _ip_host_boxes(items: list[OcrItem], frame_w: int, frame_h: int) -> list[BBox]:
    """Blur the embedded IP / sslip host, not the whole phishing URL."""
    boxes: list[BBox] = []
    for item in items:
        text = item.text or ""
        match = _IP_HOST_RE.search(text)
        if match:
            start, end = match.start(), match.end()
            if start > 0 and text[start - 1] == ".":
                start -= 1
        else:
            match = re.search(r"\.?(?:sslip|nip|xip)\.io", text, re.I)
            if not match:
                continue
            start, end = match.start(), match.end()
        box = slice_bbox(item.bbox, max(1, len(text)), start, end)
        pad_l = 18
        pad_r = max(40, int(item.bbox.w * 0.10))
        pad_v = 8
        x = max(0, box.x - pad_l)
        y = max(0, box.y - pad_v)
        w = min(frame_w - x, max(150, box.w + pad_l + pad_r))
        h = min(frame_h - y, box.h + 2 * pad_v)
        boxes.append(BBox(x, y, w, h))
    return boxes


def _looks_like_public_url(token: str) -> bool:
    if "@" in token.split("://", 1)[-1].split("/", 1)[0]:
        return False
    return bool(re.search(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,24}(?:/|$)", token))


def _group_rows(items: list[OcrItem], y_tol: int = 14) -> list[list[OcrItem]]:
    rows: list[list[OcrItem]] = []
    for item in sorted(items, key=lambda i: (i.bbox.y, i.bbox.x)):
        mid = item.bbox.y + item.bbox.h / 2
        placed = False
        for row in rows:
            row_mid = sum(i.bbox.y + i.bbox.h / 2 for i in row) / len(row)
            if abs(mid - row_mid) <= y_tol:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    return rows
