from __future__ import annotations

import unittest

from leaklens.config import AppConfig
from leaklens.detectors.base import FrameContext
from leaklens.detectors.heuristic import HeuristicDetector
from leaklens.detectors.regex import RegexDetector
from leaklens.models import BBox, OcrItem
from leaklens.tracking import apply_temporal_tracking
from leaklens.models import Detection


def _ctx(text: str) -> FrameContext:
    item = OcrItem(text=text, confidence=0.95, bbox=BBox(10, 10, 400, 30))
    return FrameContext(
        frame_bgr=None,
        frame_number=0,
        timestamp=0.0,
        width=640,
        height=480,
        ocr_items=[item],
        ocr_errors=[],
        config=AppConfig(),
    )


class RegexDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = RegexDetector(AppConfig())

    def _cats(self, text: str) -> set[str]:
        return {d.category for d in self.detector.detect(_ctx(text))}

    def test_private_and_public_ips(self) -> None:
        self.assertIn("ipv4_private", self._cats("ssh root@192.168.1.20"))
        self.assertIn("ipv4_public", self._cats("connected to 8.8.8.8"))

    def test_user_at_host_and_email(self) -> None:
        self.assertIn("hostname", self._cats("root@app-server-01: ~"))
        self.assertIn("email", self._cats("contact admin@example.com for help"))

    def test_paths_and_mac(self) -> None:
        self.assertIn("windows_path", self._cats(r"C:\Users\alex\.ssh\id_ed25519"))
        self.assertIn("unix_home_path", self._cats("cat /home/alex/.bashrc"))
        self.assertIn("mac_address", self._cats("ether 00:1A:2B:3C:4D:5E"))

    def test_secrets(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.abcabcabcabcabcabcabca"
        self.assertIn("jwt", self._cats(jwt))
        self.assertIn("cloud_credential", self._cats("AKIAIOSFODNN7EXAMPLE"))
        self.assertIn("database_connection", self._cats("postgres://user:pass@10.0.0.4:5432/app"))
        self.assertIn("ssh_key", self._cats("-----BEGIN OPENSSH PRIVATE KEY-----"))

    def test_ascii_art_at_signs_are_not_handles(self) -> None:
        art = "@@@@@@@@######++++++++........"
        self.assertNotIn("username_handle", self._cats(art))
        self.assertNotIn("hostname", self._cats("O@QQ @@C0C00C0@000"))

    def test_full_name_ignores_dates(self) -> None:
        self.assertNotIn("full_name", self._cats("Fri Aug 15 12:00:00 UTC 2025"))
        self.assertNotIn("full_name", self._cats("Welcome Welcome"))
        self.assertNotIn("full_name", self._cats("Ubuntu Linux"))
        self.assertNotIn("full_name", self._cats("Download on the App Store"))
        self.assertNotIn("full_name", self._cats("GET IT ON Google Play"))
        self.assertNotIn("full_name", self._cats("Social Engineering Video Tip"))
        self.assertNotIn("full_name", self._cats("Display Capture"))
        self.assertIn("full_name", self._cats("Ada Lovelace"))

    def test_gmail_tab_username_is_redacted(self) -> None:
        cats = self._cats("M Inbox (10,652) - adalovelace")
        self.assertIn("username_handle", cats)
        dets = self.detector.detect(_ctx("Inbox (10,652) - adalovelacex"))
        handles = [d for d in dets if d.category == "username_handle"]
        self.assertTrue(handles)
        self.assertIn("adalovelace", handles[0].text)
        tab = OcrItem("adalovelace09", 0.99, BBox(1409, 12, 74, 17))
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=[tab],
            ocr_errors=[],
            config=AppConfig(),
        )
        heur = HeuristicDetector(ctx.config).detect(ctx)
        self.assertTrue(any(d.text == "adalovelace09" for d in heur))

    def test_truncated_gmail_is_still_email(self) -> None:
        self.assertIn("email", self._cats("login ada.lovelace@gmail.c"))
        self.assertIn("email", self._cats("saved ada.lovelace@gmail."))

    def test_ip_inside_sslip_host_blurs_the_host(self) -> None:
        text = "https://www.instagram.com.203.0.113.10.sslip.io/accounts/login/"
        item = OcrItem(text=text, confidence=0.99, bbox=BBox(100, 50, 420, 20))
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=[item],
            ocr_errors=[],
            config=AppConfig(bbox_padding_px=0, expand_match_to_ocr_line=False),
        )
        dets = RegexDetector(ctx.config).detect(ctx)
        ip_dets = [d for d in dets if "203.0.113.10" in (d.text or "")]
        self.assertTrue(ip_dets, [(d.category, d.text, d.bbox) for d in dets])
        from leaklens.util import slice_bbox

        ip_span = slice_bbox(item.bbox, len(text), text.index("203.0.113.10"), text.index("sslip.io") + 8)
        box = ip_dets[0].bbox
        self.assertLessEqual(box.x, ip_span.x + 20)
        self.assertGreaterEqual(box.x + box.w, ip_span.x + ip_span.w - 20)
        self.assertGreater(box.x, item.bbox.x + 30)

    def test_revealed_url_blurs_ip_not_the_whole_bar(self) -> None:
        text = "https://www.instagram.com.203.0.113.10.sslip.io/revealed?t=ExampleTokenValue99"
        item = OcrItem(text=text, confidence=0.99, bbox=BBox(80, 47, 585, 24))
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=[item],
            ocr_errors=[],
            config=AppConfig(bbox_padding_px=0, expand_match_to_ocr_line=False),
        )
        dets = RegexDetector(ctx.config).detect(ctx)
        ip_dets = [d for d in dets if "203.0.113.10" in (d.text or "")]
        self.assertTrue(ip_dets)
        box = max(ip_dets, key=lambda d: d.bbox.w).bbox
        from leaklens.util import slice_bbox

        ip_span = slice_bbox(item.bbox, len(text), text.index("203.0.113.10"), text.index("sslip.io") + 8)
        self.assertLessEqual(box.x, ip_span.x + 24)
        self.assertGreaterEqual(box.x + box.w, ip_span.x + ip_span.w - 24)
        self.assertLess(box.x + box.w, item.bbox.x + item.bbox.w - 80)

    def test_autofill_email_stack(self) -> None:
        items = [
            OcrItem("ada.lovelace@example.com", 0.99, BBox(1096, 426, 165, 16)),
            OcrItem("dev.null@example.com", 0.99, BBox(1091, 472, 127, 16)),
            OcrItem("ops.desk@example.com", 0.99, BBox(1094, 517, 168, 18)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        self.assertTrue(any(d.detector == "heuristic.email_stack" for d in dets))
        stack = next(d for d in dets if d.detector == "heuristic.email_stack")
        self.assertGreaterEqual(stack.bbox.h, 90)

    def test_saved_info_popup_covers_username_rows(self) -> None:
        items = [
            OcrItem("Saved info", 0.99, BBox(1091, 389, 67, 16)),
            OcrItem("RiverStone", 0.99, BBox(1094, 425, 140, 16)),
            OcrItem("test", 0.99, BBox(1094, 460, 40, 16)),
            OcrItem("Manage personal info", 0.99, BBox(1116, 606, 141, 19)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        popups = [d for d in dets if d.detector == "heuristic.identity_popup"]
        self.assertTrue(popups)
        box = popups[0].bbox
        self.assertLessEqual(box.y, 389)
        self.assertGreaterEqual(box.y + box.h, 476)
        self.assertTrue(any("RiverStone" in (d.text or "") for d in popups))

    def test_saved_info_popup_covers_all_emails(self) -> None:
        items = [
            OcrItem("Saved info", 0.99, BBox(1091, 389, 67, 16)),
            OcrItem("ada.lovelace@example.com", 0.99, BBox(1096, 426, 165, 16)),
            OcrItem("dev.null@example.com", 0.99, BBox(1091, 472, 127, 16)),
            OcrItem("ops.desk@example.com", 0.99, BBox(1094, 517, 168, 18)),
            OcrItem("casey.ng@example.com", 0.99, BBox(1091, 562, 152, 20)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        popups = [d for d in dets if d.detector == "heuristic.identity_popup"]
        self.assertTrue(popups)
        box = popups[0].bbox
        self.assertGreaterEqual(box.h, 160)
        self.assertGreaterEqual(box.w, 220)

    def test_dark_panel_expand_accepts_video_frames(self) -> None:
        import numpy as np

        from leaklens.detectors.heuristic import expand_to_dark_panel

        frame = np.full((180, 320, 3), 240, dtype=np.uint8)
        frame[40:140, 80:240] = 30
        box = BBox(100, 60, 80, 40)
        expanded = expand_to_dark_panel(frame, box, 320, 180)
        self.assertGreaterEqual(expanded.w, box.w)
        self.assertGreaterEqual(expanded.h, box.h)

    def test_obs_chrome_is_not_an_identity_popup(self) -> None:
        items = [
            OcrItem("app-server-01·Host", 0.99, BBox(488, 10, 163, 24)),
            OcrItem("Scenes", 0.99, BBox(520, 40, 48, 16)),
            OcrItem("Scene", 0.99, BBox(520, 200, 40, 16)),
            OcrItem("Scenes", 0.99, BBox(520, 220, 48, 16)),
            OcrItem("Fade", 0.99, BBox(40, 900, 36, 16)),
            OcrItem("Duration", 0.99, BBox(90, 900, 70, 16)),
            OcrItem("TextNow", 0.99, BBox(1110, 8, 46, 23)),
            OcrItem("Tools", 0.99, BBox(80, 860, 40, 16)),
            OcrItem("Help", 0.99, BBox(130, 860, 36, 16)),
            OcrItem("Instagram", 0.99, BBox(1080, 210, 90, 28)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        self.assertFalse(any(d.detector == "heuristic.identity_popup" for d in dets), [d.text for d in dets])

    def test_username_rows_still_cover_the_login_field(self) -> None:
        items = [
            OcrItem("Saved info", 0.99, BBox(1091, 389, 67, 16)),
            OcrItem("RiverStone", 0.99, BBox(1094, 425, 140, 16)),
            OcrItem("test", 0.99, BBox(1094, 460, 40, 16)),
            OcrItem("ada.lovelace@example.com", 0.99, BBox(1040, 343, 160, 20)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        popups = [d for d in dets if d.detector == "heuristic.identity_popup"]
        self.assertTrue(popups)
        box = popups[0].bbox
        self.assertLessEqual(box.y, 343)
        self.assertGreaterEqual(box.y + box.h, 476)

    def test_split_sslip_url_blurs_the_ip_and_wildcard(self) -> None:
        items = [
            OcrItem("https://www.instagram.com.", 0.99, BBox(108, 51, 180, 19)),
            OcrItem("203.0.113.10", 0.99, BBox(288, 51, 90, 19)),
            OcrItem(".sslip.io/accounts/login/", 0.99, BBox(378, 51, 140, 19)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(bbox_padding_px=0, expand_match_to_ocr_line=False),
        )
        dets = RegexDetector(ctx.config).detect(ctx)
        covered = [d for d in dets if d.bbox.x <= 300 and d.bbox.x + d.bbox.w >= 430]
        self.assertTrue(covered, [(d.category, d.text, d.bbox) for d in dets])

    def test_url_is_not_a_mail_tab_username(self) -> None:
        item = OcrItem(
            "https://www.instagram.com.203.0.113.10.sslip.io/accounts/login/",
            0.99,
            BBox(108, 51, 413, 19),
        )
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=[item],
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        self.assertFalse(any(d.detector == "heuristic.mail_tab" for d in dets))

    def test_phishing_results_page_is_not_autofill(self) -> None:
        items = [
            OcrItem("Instagram", 0.99, BBox(1080, 210, 90, 28)),
            OcrItem("What the attacker captured", 0.99, BBox(731, 499, 203, 18)),
            OcrItem("TEST", 0.99, BBox(736, 576, 35, 22)),
            OcrItem("PASSWORD", 0.99, BBox(728, 603, 78, 21)),
            OcrItem("TEST", 0.99, BBox(737, 626, 32, 20)),
            OcrItem("Captured at 2026-08-14T09:30:05.120252+00:00 UTC", 0.99, BBox(731, 657, 298, 18)),
            OcrItem("Check the URL — it wasn't instagram.com", 0.99, BBox(730, 751, 295, 19)),
            OcrItem("Use a password manager — it won't autofill on fake sites", 0.99, BBox(730, 827, 391, 18)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        self.assertFalse(any(d.category == "autofill_suggestion" for d in dets), [d.text for d in dets])
        logo = BBox(1080, 210, 90, 28)
        overlapping = [d for d in dets if d.bbox.iou(logo) > 0.2]
        self.assertFalse(overlapping, [(d.detector, d.text, d.bbox) for d in overlapping])

    def test_password_prompt_is_not_a_value(self) -> None:
        prompt = "root@8.8.8.8's password:"
        cats = self._cats(prompt)
        self.assertNotIn("password", cats)
        self.assertIn("ipv4_public", cats)
        word = OcrItem("root@8.8.8.8's", 0.99, BBox(10, 40, 200, 20))
        label = OcrItem("password:", 0.99, BBox(220, 40, 90, 20))
        line = OcrItem("root@8.8.8.8's password:", 0.98, BBox(10, 40, 310, 20))
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=640,
            height=480,
            ocr_items=[word, label, line],
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = RegexDetector(ctx.config).detect(ctx)
        self.assertFalse(any(d.category == "password" for d in dets), [d.text for d in dets])
        self.assertTrue(any(d.category == "ipv4_public" for d in dets))

    def test_password_value_is_redacted(self) -> None:
        dets = self.detector.detect(_ctx("password: hunter2"))
        values = [d for d in dets if d.category == "password"]
        self.assertTrue(values)
        self.assertEqual(values[0].text, "hunter2")
        self.assertEqual(values[0].classification, "SENSITIVE")
        self.assertIn("password", self._cats("PASSWORD=s3cretvalue"))

    def test_motd_stack_is_not_autofill(self) -> None:
        items = [
            OcrItem("*** System restart required ***", 0.99, BBox(40, 100, 280, 18)),
            OcrItem("You have new mail.", 0.99, BBox(42, 122, 270, 18)),
            OcrItem("root@app-server-01:~#", 0.99, BBox(44, 144, 260, 18)),
        ]
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=640,
            height=480,
            ocr_items=items,
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = HeuristicDetector(ctx.config).detect(ctx)
        self.assertFalse(any(d.category == "autofill_suggestion" for d in dets))

    def test_ordinary_commands_are_not_redacted(self) -> None:
        cats = self._cats("apt install curl && ssh")
        self.assertNotIn("terminal_secret", cats)
        self.assertFalse(any(d.detector == "regex.keyword_context" for d in self.detector.detect(_ctx("apt install curl && ssh"))))

    def test_blurs_only_the_ip_span(self) -> None:
        text = "curl https://192.168.1.50/api"
        item = OcrItem(text=text, confidence=0.99, bbox=BBox(0, 10, 280, 20))
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=640,
            height=480,
            ocr_items=[item],
            ocr_errors=[],
            config=AppConfig(bbox_padding_px=0, expand_match_to_ocr_line=False),
        )
        dets = RegexDetector(ctx.config).detect(ctx)
        ip_dets = [d for d in dets if d.category == "ipv4_private"]
        self.assertTrue(ip_dets)
        box = ip_dets[0].bbox
        self.assertGreater(box.x, 40)
        self.assertLess(box.x + box.w, 260)
        self.assertEqual(ip_dets[0].classification, "SENSITIVE")
        self.assertEqual(ip_dets[0].origin, "direct")
        title = OcrItem("root@app-server-01:~", 0.99, BBox(10, 2, 240, 28))
        prompt = OcrItem("root@1.2.3.4's password:", 0.99, BBox(10, 640, 320, 28))
        ctx = FrameContext(
            frame_bgr=None,
            frame_number=0,
            timestamp=0.0,
            width=1920,
            height=1080,
            ocr_items=[title, prompt],
            ocr_errors=[],
            config=AppConfig(),
        )
        dets = self.detector.detect(ctx)
        for det in dets:
            self.assertLess(det.bbox.h, 120, det)


class TrackingTests(unittest.TestCase):
    def test_fills_short_gaps(self) -> None:
        config = AppConfig(track_max_gap_frames=3, track_iou_threshold=0.2)
        a = Detection(0, 0.0, BBox(10, 10, 50, 20), "hostname", "regex", 0.9, text="root@host")
        b = Detection(2, 0.06, BBox(12, 11, 50, 20), "hostname", "regex", 0.9, text="root@host")
        out = apply_temporal_tracking([a, b], config, fps=30)
        frames = {d.frame_number for d in out}
        self.assertEqual(frames, {0, 1, 2})
        self.assertTrue(any(d.temporal_extended for d in out))

    def test_single_frame_leak_is_kept(self) -> None:
        config = AppConfig(track_max_gap_frames=8)
        only = Detection(7, 0.23, BBox(10, 10, 50, 20), "ipv4_private", "regex.ipv4", 0.9, text="10.0.0.8")
        out = apply_temporal_tracking([only], config, fps=30, frame_count=20)
        self.assertIn(7, {d.frame_number for d in out})
        self.assertTrue(any(not d.temporal_extended for d in out))

    def test_hold_when_neighbor_ocr_unreadable(self) -> None:
        config = AppConfig(track_max_gap_frames=3, track_hold_unreadable_frames=3)
        hit = Detection(1, 0.03, BBox(10, 10, 80, 20), "ipv4_private", "regex.ipv4", 0.9, text="10.1.2.3")
        ocr = {
            1: [OcrItem("10.1.2.3", 0.9, BBox(10, 10, 80, 20))],
            2: [],
        }
        out = apply_temporal_tracking([hit], config, fps=30, ocr_by_frame=ocr, frame_count=5)
        self.assertIn(2, {d.frame_number for d in out})


if __name__ == "__main__":
    unittest.main()
