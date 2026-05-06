"""Browser-based region reviewer.

Spins up a local HTTP server, opens a page that embeds YouTube's IFrame
player. Keystrokes captured in the page query the player's current time
to record regions, then POST them to the server which writes a regions
YAML.  No video download needed.
"""
import json
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from carai.data.regions import Region, read_regions, regions_path_for, write_regions


def _video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.lstrip("/")
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    raise ValueError(f"cannot extract video id from {url!r}")


def build_queue(sources_yaml: Path, regions_dir: Path) -> list[dict]:
    sources = yaml.safe_load(Path(sources_yaml).read_text())
    out: list[dict] = []
    for car, urls in sources.items():
        for url in urls:
            try:
                vid = _video_id_from_url(url)
            except ValueError:
                continue
            rpath = regions_path_for(regions_dir, car, vid)
            reviewed = False
            if rpath.exists():
                try:
                    _, _, regs = read_regions(rpath)
                    reviewed = bool(regs)
                except Exception:
                    pass
            out.append({
                "car": car,
                "video_id": vid,
                "url": url,
                "reviewed": reviewed,
            })
    return out


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>carAI region reviewer</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #111; color: #e8e8e8;
         margin: 0; padding: 20px; max-width: 940px; }
  h1 { font-size: 16px; margin: 0 0 6px 0; letter-spacing: 0.02em; }
  .meta { font-size: 13px; opacity: 0.7; margin-bottom: 12px; }
  #player-wrap { position: relative; width: 854px; height: 480px; background: #000; }
  .focus-badge { position: absolute; top: 8px; right: 8px; padding: 4px 10px; font-size: 11px;
                 background: rgba(220,80,80,0.9); border-radius: 4px; cursor: pointer;
                 display: none; user-select: none; }
  .focus-badge.show { display: block; }
  .row { margin-top: 14px; display: flex; gap: 18px; align-items: center; }
  .mode { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 13px;
          font-weight: 600; letter-spacing: 0.04em; }
  .mode.idle  { background: #2c5d8f; }
  .mode.accel { background: #b04a1a; }
  .mode.decel { background: #1a7a6a; }
  .mode.skip  { background: #444; }
  .mode.none  { background: #2a2a2a; opacity: 0.6; }
  .time { font-family: ui-monospace, monospace; font-size: 13px; opacity: 0.85; }
  .keys { font-size: 12px; opacity: 0.65; line-height: 1.6; margin-top: 8px; }
  .keys b { color: #fff; opacity: 1; }
  .ranges { font-family: ui-monospace, monospace; font-size: 12px; margin-top: 12px;
            max-height: 200px; overflow-y: auto; border: 1px solid #2a2a2a; padding: 8px;
            background: #181818; border-radius: 4px; }
  .ranges .r { padding: 2px 0; }
  button { background: #2a2a2a; color: #eee; border: 1px solid #3a3a3a; padding: 6px 14px;
           border-radius: 4px; cursor: pointer; font-size: 13px; }
  button:hover { background: #353535; }
  .done { padding: 40px; text-align: center; font-size: 16px; }
</style></head><body>
<h1>carAI region reviewer</h1>
<div class="meta" id="meta">loading...</div>
<div id="player-wrap">
  <div id="player"></div>
  <div class="focus-badge" id="focus-badge" onclick="document.body.focus()">click to refocus keys</div>
</div>
<div class="row">
  <span>mode: <span id="current-mode" class="mode none">none</span></span>
  <span class="time" id="time">0.0 / 0.0s</span>
  <button id="next-btn" onclick="saveAndNext()">save & next (n)</button>
  <button onclick="loadNext(false)">skip without save (S)</button>
</div>
<div class="keys">
  <b>i</b> idle &middot; <b>a</b> accel &middot; <b>d</b> decel &middot; <b>s</b> skip &middot;
  <b>x</b> close current &middot; <b>u</b> undo &middot;
  <b>n</b> save &amp; next &middot; <b>S</b> next without save
  <br>YouTube controls: <b>space</b> play/pause &middot; <b>k</b> play/pause &middot;
  <b>j/l</b> -10/+10s &middot; <b>0-9</b> jump to %
</div>
<div class="ranges" id="ranges"></div>
<script src="https://www.youtube.com/iframe_api"></script>
<script>
let player = null, queue = [], idx = -1;
let currentMode = null, modeStart = 0, ranges = [];

document.body.tabIndex = 0;  // allow body to receive focus

fetch('/api/queue').then(r => r.json()).then(q => {
  queue = q.filter(v => !v.reviewed);
  if (queue.length === 0) {
    document.body.innerHTML = '<div class="done">all videos already reviewed.</div>';
    return;
  }
  loadNext(false);
});

window.onYouTubeIframeAPIReady = function() {
  // first call defers until queue is loaded; idx<0 means waiting
};

function ensurePlayer(videoId, onReady) {
  if (player && player.loadVideoById) {
    player.loadVideoById(videoId);
    onReady();
    return;
  }
  player = new YT.Player('player', {
    height: '480', width: '854', videoId,
    playerVars: { rel: 0, modestbranding: 1 },
    events: { onReady: onReady,
              onStateChange: () => { /* nothing for now */ } }
  });
}

function loadNext(autoIncrement = true) {
  if (autoIncrement) idx++;
  else idx++;  // always advance; the "skip without save" button calls loadNext(false) but still advances
  if (idx >= queue.length) {
    document.getElementById('meta').textContent = 'done';
    if (player) try { player.stopVideo(); } catch(e){}
    return;
  }
  const v = queue[idx];
  document.getElementById('meta').textContent =
    '[' + (idx+1) + '/' + queue.length + ']  ' + v.car + ' / ' + v.video_id +
    '  (' + (queue.length - idx - 1) + ' remaining)';
  ranges = []; currentMode = null; modeStart = 0;
  renderRanges();
  setModeBadge(null);
  ensurePlayer(v.video_id, () => {
    document.body.focus();
    refreshFocusBadge();
  });
}

function setModeBadge(label) {
  const el = document.getElementById('current-mode');
  el.textContent = label || 'none';
  el.className = 'mode ' + (label || 'none');
}

function setMode(newMode) {
  if (!player || !player.getCurrentTime) return;
  const now = player.getCurrentTime();
  if (currentMode !== null && now > modeStart + 0.05) {
    ranges.push({start: modeStart, end: now, label: currentMode});
  }
  currentMode = newMode;
  modeStart = now;
  setModeBadge(newMode);
  renderRanges();
}

function renderRanges() {
  const html = ranges.map((r, i) =>
    '<div class="r">[' + (i+1) + '] ' + r.start.toFixed(1) + '–' + r.end.toFixed(1) +
    's  <span class="mode ' + r.label + '">' + r.label + '</span>  (' +
    (r.end - r.start).toFixed(1) + 's)</div>'
  ).join('');
  document.getElementById('ranges').innerHTML = html || '<div style="opacity:.5">no ranges yet</div>';
}

function saveAndNext() {
  if (!player || !player.getCurrentTime) return;
  const v = queue[idx];
  const duration = player.getDuration();
  if (currentMode !== null) {
    const now = player.getCurrentTime();
    if (now > modeStart + 0.05) ranges.push({start: modeStart, end: now, label: currentMode});
    currentMode = null;
  }
  fetch('/api/regions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({car: v.car, video_id: v.video_id, duration, ranges})
  }).then(r => r.json()).then(_ => loadNext(true));
}

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const k = e.key;
  if (k === 'i') { setMode('idle'); e.preventDefault(); }
  else if (k === 'a') { setMode('accel'); e.preventDefault(); }
  else if (k === 'd') { setMode('decel'); e.preventDefault(); }
  else if (k === 's') { setMode('skip'); e.preventDefault(); }
  else if (k === 'x') { setMode(null); e.preventDefault(); }
  else if (k === 'u' && ranges.length) {
    const last = ranges.pop();
    currentMode = last.label;
    modeStart = last.start;
    setModeBadge(last.label);
    renderRanges();
    e.preventDefault();
  }
  else if (k === 'n') { saveAndNext(); e.preventDefault(); }
  else if (k === 'S') { loadNext(true); e.preventDefault(); }
});

function refreshFocusBadge() {
  const badge = document.getElementById('focus-badge');
  // We can't reliably detect iframe focus; show badge when document.activeElement is the iframe
  setTimeout(() => {
    const iframe = document.querySelector('#player iframe');
    if (document.activeElement === iframe) badge.classList.add('show');
    else badge.classList.remove('show');
  }, 100);
}
window.addEventListener('focus', refreshFocusBadge, true);
window.addEventListener('blur', refreshFocusBadge, true);
setInterval(refreshFocusBadge, 1000);

setInterval(() => {
  if (player && player.getCurrentTime && player.getDuration) {
    const t = player.getCurrentTime();
    const d = player.getDuration();
    document.getElementById('time').textContent = t.toFixed(1) + ' / ' + d.toFixed(1) + 's';
  }
}, 200);
</script></body></html>
"""


def _make_handler(sources_yaml: Path, regions_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/api/queue":
                queue = build_queue(sources_yaml, regions_dir)
                self._send(200, "application/json", json.dumps(queue).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path == "/api/regions":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw)
                    car = data["car"]
                    video_id = data["video_id"]
                    duration = float(data.get("duration") or 0.0)
                    regions = []
                    for r in data.get("ranges", []):
                        try:
                            regions.append(Region(float(r["start"]), float(r["end"]), r["label"]))
                        except (KeyError, ValueError):
                            continue
                    rpath = regions_path_for(regions_dir, car, video_id)
                    write_regions(rpath, video_id=video_id, duration=duration, regions=regions)
                    self._send(200, "application/json",
                               json.dumps({"ok": True, "saved": str(rpath)}).encode())
                except Exception as e:
                    self._send(400, "application/json",
                               json.dumps({"ok": False, "error": str(e)}).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def log_message(self, fmt: str, *args) -> None:
            pass  # quiet

    return Handler


def _find_free_port(prefer: int = 7777) -> int:
    for port in [prefer, *range(7780, 7800)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError("no free port found")


def serve(
    sources_yaml: Path,
    regions_dir: Path,
    port: int = 7777,
    open_browser: bool = True,
) -> None:
    port = _find_free_port(port)
    handler = _make_handler(sources_yaml, regions_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n  carAI web reviewer: {url}\n  (Ctrl-C to stop)\n")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
        server.shutdown()
