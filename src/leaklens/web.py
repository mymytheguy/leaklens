"""Local upload UI for photos and videos. Processing stays on this machine."""

from __future__ import annotations

import json
import mimetypes
import tempfile
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from leaklens.config import AppConfig
from leaklens.exceptions import LeakLensError
from leaklens.scanner import run_scan
from leaklens.video_io import IMAGE_SUFFIXES, VIDEO_SUFFIXES, is_image_path

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def serve_app(*, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"LeakLens  {url}")
    print("Uploads stay on this machine. Ctrl+C to stop.")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("Warning: this page is reachable from other machines.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
    return 0


class _Handler(BaseHTTPRequestHandler):
    server_version = "LeakLens/0.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
        if parsed.path.startswith("/job/"):
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) == 2:
                return self._job_json(parts[1])
            if len(parts) == 3 and parts[2] in {"file", "review"}:
                return self._job_file(parts[1], parts[2])
        self._send(404, "application/json", b'{"error":"not found"}')

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/scan":
            self._send(404, "application/json", b'{"error":"not found"}')
            return
        try:
            name, data = _read_upload(self)
        except ValueError as exc:
            self._send(400, "application/json", json.dumps({"error": str(exc)}).encode())
            return
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            self._send(415, "application/json", b'{"error":"upload a photo or video"}')
            return
        job_id = uuid.uuid4().hex[:12]
        work = Path(tempfile.mkdtemp(prefix=f"leaklens_{job_id}_"))
        source = work / Path(name).name
        source.write_bytes(data)
        dest = work / f"{source.stem}.redacted{source.suffix.lower() if is_image_path(source) else '.mp4'}"
        report = work / f"{source.stem}.leaklens"
        with LOCK:
            JOBS[job_id] = {"status": "running", "name": source.name}
        thread = threading.Thread(
            target=_run_job,
            args=(job_id, source, dest, report),
            daemon=True,
        )
        thread.start()
        self._send(202, "application/json", json.dumps({"id": job_id}).encode())

    def _job_json(self, job_id: str) -> None:
        with LOCK:
            job = JOBS.get(job_id)
        if not job:
            self._send(404, "application/json", b'{"error":"unknown job"}')
            return
        self._send(200, "application/json", json.dumps(job, default=str).encode())

    def _job_file(self, job_id: str, kind: str) -> None:
        with LOCK:
            job = JOBS.get(job_id)
        if not job or job.get("status") != "done":
            self._send(404, "application/json", b'{"error":"not ready"}')
            return
        path = Path(job["redacted"] if kind == "file" else job["review"])
        if not path.is_file():
            self._send(404, "application/json", b'{"error":"missing file"}')
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _run_job(job_id: str, source: Path, dest: Path, report: Path) -> None:
    try:
        result = run_scan(
            source,
            output_video=dest,
            report_dir=report,
            config=AppConfig(),
            print_summary=False,
        )
        summary = result.summary
        payload = {
            "status": "done",
            "name": source.name,
            "kind": "image" if is_image_path(source) else "video",
            "redacted": str(result.outputs["redacted_preview"]),
            "review": str(result.outputs["review"]),
            "tracks": summary.get("track_count", 0),
            "sensitive": summary.get("sensitive", 0),
            "review_required": summary.get("review_required", 0),
            "categories": summary.get("by_category") or {},
            "width": result.video_info.width,
            "height": result.video_info.height,
        }
    except LeakLensError as exc:
        payload = {"status": "error", "error": str(exc), "name": source.name}
    except Exception as exc:
        payload = {"status": "error", "error": f"scan failed: {exc}", "name": source.name}
    with LOCK:
        JOBS[job_id] = payload


def _read_upload(handler: BaseHTTPRequestHandler) -> tuple[str, bytes]:
    ctype = handler.headers.get("Content-Type") or ""
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0 or length > MAX_UPLOAD_BYTES:
        raise ValueError("file is missing or too large")
    body = handler.rfile.read(length)
    if "multipart/form-data" not in ctype:
        raise ValueError("expected a file upload")
    boundary = ""
    for part in ctype.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
    if not boundary:
        raise ValueError("malformed upload")
    token = b"--" + boundary.encode("ascii", "ignore")
    chunks = body.split(token)
    for chunk in chunks:
        header, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep or b"filename=" not in header:
            continue
        line = header.decode("utf-8", "replace")
        name = ""
        for piece in line.split(";"):
            piece = piece.strip()
            if piece.lower().startswith("filename="):
                name = unquote(piece.split("=", 1)[1].strip().strip('"'))
        if not name:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        return Path(name).name, data
    raise ValueError("no file in upload")


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeakLens</title>
<style>
:root{--bg:#0e1116;--card:#171b22;--line:#2a3140;--fg:#e8eaed;--mute:#8b93a7;--ok:#81c995;--warn:#f4b183}
*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;background:var(--bg);color:var(--fg);font:15px/1.45 ui-sans-serif,system-ui,sans-serif;padding:32px 18px 48px}
main{max-width:720px;margin:0 auto}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.brand svg{width:22px;height:22px}
h1{font-size:22px;font-weight:650;letter-spacing:-.02em}
.lead{color:var(--mute);margin:0 0 22px;font-size:14px}
.drop{border:1px dashed #3a4356;background:var(--card);border-radius:16px;padding:36px 20px;text-align:center;cursor:pointer}
.drop:hover,.drop.on{border-color:#6b8cff;background:#1b2130}
.icons{display:flex;justify-content:center;gap:18px;margin-bottom:12px}
.icons figure{margin:0;color:var(--mute);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
.icons svg{width:28px;height:28px;display:block;margin:0 auto 6px;stroke:currentColor;fill:none;stroke-width:1.6}
.drop strong{display:block;margin-bottom:4px}
.drop span{color:var(--mute);font-size:13px}
input[type=file]{display:none}
.status{margin-top:16px;color:var(--mute);font-size:13px}
.result{margin-top:22px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 14px}
.chip{border:1px solid var(--line);border-radius:999px;padding:4px 10px;font-size:12px;color:var(--mute)}
.chip b{color:var(--fg);font-weight:600}
.cats{display:grid;gap:6px;margin:0 0 14px}
.cat{display:flex;justify-content:space-between;font-size:13px;color:var(--mute);border-bottom:1px solid #222833;padding:6px 0}
.cat b{color:var(--fg);font-weight:550}
.preview{width:100%;border-radius:10px;background:#000;margin:8px 0 14px;max-height:360px;object-fit:contain}
.actions{display:flex;gap:8px;flex-wrap:wrap}
a.btn{color:var(--fg);text-decoration:none;border:1px solid var(--line);border-radius:10px;padding:8px 12px;font-size:13px}
a.btn.primary{background:#243056;border-color:#3d5080}
.err{color:#f28b82}
</style>
</head>
<body>
<main>
  <div class="brand">
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7" stroke="currentColor" fill="none" stroke-width="1.6"/><circle cx="12" cy="12" r="2.4" fill="currentColor"/><path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21" stroke="currentColor" stroke-width="1.6" fill="none"/></svg>
    <h1>LeakLens</h1>
  </div>
  <p class="lead">Drop a photo or video. It is scanned on this computer and a redacted copy is written beside a short review.</p>
  <label class="drop" id="drop">
    <div class="icons">
      <figure>
        <svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="10" r="1.4" fill="currentColor" stroke="none"/><path d="M7 16l4.2-4.2L14 14.6 17 11l4 5"/></svg>
        Photo
      </figure>
      <figure>
        <svg viewBox="0 0 24 24"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10.2l5-3.2v10.2l-5-3.2z"/></svg>
        Video
      </figure>
    </div>
    <strong>Drop a file here</strong>
    <span>PNG, JPG, WebP, GIF, MP4, MOV, MKV, WebM</span>
    <input id="file" type="file" accept="image/*,video/*,.png,.jpg,.jpeg,.webp,.gif,.mp4,.mov,.mkv,.webm">
  </label>
  <p class="status" id="status"></p>
  <section class="result" id="result" hidden></section>
</main>
<script>
const drop = document.getElementById('drop');
const input = document.getElementById('file');
const status = document.getElementById('status');
const result = document.getElementById('result');

drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('on'); });
drop.addEventListener('dragleave', () => drop.classList.remove('on'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('on');
  if (e.dataTransfer.files[0]) scan(e.dataTransfer.files[0]);
});
input.addEventListener('change', () => { if (input.files[0]) scan(input.files[0]); });

async function scan(file) {
  result.hidden = true;
  status.textContent = 'Uploading ' + file.name + '…';
  const body = new FormData();
  body.append('file', file, file.name);
  const started = await fetch('/scan', { method: 'POST', body });
  const job = await started.json();
  if (!started.ok) { status.innerHTML = '<span class="err">' + (job.error || 'upload failed') + '</span>'; return; }
  status.textContent = 'Scanning locally…';
  let info;
  for (let i = 0; i < 3600; i++) {
    await new Promise(r => setTimeout(r, 700));
    const res = await fetch('/job/' + job.id);
    info = await res.json();
    if (info.status !== 'running') break;
  }
  if (!info || info.status !== 'done') {
    status.innerHTML = '<span class="err">' + ((info && info.error) || 'scan failed') + '</span>';
    return;
  }
  status.textContent = '';
  const cats = Object.entries(info.categories || {});
  const catHtml = cats.length
    ? cats.map(([name, n]) => '<div class="cat"><span>' + name.replace(/_/g,' ') + '</span><b>' + n + '</b></div>').join('')
    : '<div class="cat"><span>No classified leaks</span><b>0</b></div>';
  const preview = info.kind === 'image'
    ? '<img class="preview" alt="redacted preview" src="/job/' + job.id + '/file">'
    : '';
  result.hidden = false;
  result.innerHTML =
    '<div>' + info.name + ' · ' + info.width + '×' + info.height + ' · ' + info.kind + '</div>' +
    '<div class="stats">' +
      '<span class="chip"><b>' + info.tracks + '</b> tracks</span>' +
      '<span class="chip"><b>' + info.sensitive + '</b> sensitive</span>' +
      '<span class="chip"><b>' + info.review_required + '</b> review</span>' +
    '</div>' +
    '<div class="cats">' + catHtml + '</div>' +
    preview +
    '<div class="actions">' +
      '<a class="btn primary" href="/job/' + job.id + '/file" download>Download redacted</a>' +
      '<a class="btn" href="/job/' + job.id + '/review" target="_blank">Open review</a>' +
    '</div>';
}
</script>
</body>
</html>
"""
