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

from carzam.data.regions import Region, read_regions, regions_path_for, write_regions


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
  /* Block all pointer events on the iframe so clicks never steal focus.
     The user controls playback via the transport bar below. */
  #player-wrap iframe { pointer-events: none; }
  .transport { display: flex; gap: 6px; align-items: center; margin-top: 8px;
               padding: 6px 10px; background: #1a1a1a; border-radius: 4px; }
  .transport button { background: #2a2a2a; border: 1px solid #3a3a3a; color: #eee;
                      padding: 6px 12px; border-radius: 4px; font-size: 13px;
                      cursor: pointer; min-width: 40px; }
  .transport button:hover { background: #353535; }
  .transport .scrubber { flex: 1; margin: 0 8px; }
  .focus-badge { display: none; }  /* no longer needed */
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
  button:focus { outline: none; }  /* avoid persistent focus on transport buttons */
  body.loading { cursor: wait; }
  .loading-banner { position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
                    background: #b04a1a; color: #fff; padding: 5px 12px; border-radius: 4px;
                    font-size: 12px; display: none; z-index: 100; }
  body.loading .loading-banner { display: block; }
  .done { padding: 40px; text-align: center; font-size: 16px; }
</style></head><body>
<div class="loading-banner">loading next video — keystrokes ignored…</div>
<h1>carAI region reviewer</h1>
<div class="meta" id="meta">loading...</div>
<div id="player-wrap">
  <div id="player"></div>
</div>
<div class="transport">
  <button onclick="seekRel(-10)" title="-10s (j)">◀◀</button>
  <button onclick="seekRel(-5)" title="-5s (←)">◀</button>
  <button id="play-btn" onclick="togglePlay()" title="play/pause (space)">▶</button>
  <button onclick="seekRel(5)" title="+5s (→)">▶</button>
  <button onclick="seekRel(10)" title="+10s (l)">▶▶</button>
  <input type="range" class="scrubber" id="scrubber" min="0" max="1000" value="0" step="1">
  <button onclick="toggleMute()" title="mute (m)" id="mute-btn">🔊</button>
</div>
<div class="row">
  <span>mode: <span id="current-mode" class="mode none">none</span></span>
  <span class="time" id="time">0.0 / 0.0s</span>
  <span id="quick-status" style="font-family: monospace; font-size: 12px; opacity: 0.85;">range: — / —</span>
  <button id="next-btn" onclick="saveAndNext()">save & next (n)</button>
  <button onclick="loadNext(true)">skip without save (S)</button>
</div>
<div class="keys">
  Quick approve: <b>b</b> mark whole video &amp; advance &middot;
  <b>[</b> mark range-start at current time &middot;
  <b>]</b> commit range start..now and continue (repeat for multiple segments) &middot;
  <b>e</b> save all collected ranges &amp; advance
  <br>Per-state labels (advanced): <b>i</b> idle &middot; <b>a</b> accel &middot; <b>d</b> decel &middot; <b>s</b> skip &middot;
  <b>x</b> close current &middot; <b>u</b> undo &middot; <b>n</b> save state-labeled &amp; next
  <br>Skip / playback: <b>S</b> next without save &middot; <b>space</b> play/pause &middot;
  <b>j</b>/<b>l</b> &plusmn;10s &middot; <b>←</b>/<b>→</b> &plusmn;5s &middot; <b>m</b> mute
</div>
<div class="ranges" id="ranges"></div>
<script src="https://www.youtube.com/iframe_api"></script>
<script>
let player = null, queue = [], idx = -1;
let currentMode = null, modeStart = 0, ranges = [];
let playerReady = false;  // gates keystrokes during async video loads

document.body.tabIndex = 0;  // allow body to receive focus

// Helper: any time a button is clicked, drop focus back to body
// so the next keypress reaches our handler instead of the button.
document.addEventListener('click', (e) => {
  if (e.target.tagName === 'BUTTON') {
    setTimeout(() => { e.target.blur(); document.body.focus(); }, 0);
  }
}, true);

function setLoading(on) {
  playerReady = !on;
  document.body.classList.toggle('loading', on);
}

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
  setLoading(true);
  const markReady = () => { setLoading(false); onReady && onReady(); };
  if (player && player.loadVideoById) {
    player.loadVideoById(videoId);
    // loadVideoById doesn't refire onReady; wait until duration is known.
    let tries = 0;
    const wait = setInterval(() => {
      tries++;
      try {
        if (player.getDuration && player.getDuration() > 0) {
          clearInterval(wait);
          markReady();
        } else if (tries > 60) {  // 6s timeout
          clearInterval(wait);
          markReady();  // proceed anyway; button checks will guard
        }
      } catch (e) {}
    }, 100);
    return;
  }
  player = new YT.Player('player', {
    height: '480', width: '854', videoId,
    playerVars: { rel: 0, modestbranding: 1 },
    events: { onReady: markReady, onStateChange: () => {} }
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
  quickStart = null; quickEnd = null;
  renderRanges();
  setModeBadge(null);
  if (typeof updateQuickStatus === 'function') updateQuickStatus();
  ensurePlayer(v.video_id, () => {
    document.body.focus();
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

function togglePlay() {
  if (!player || !player.getPlayerState) return;
  const state = player.getPlayerState();
  if (state === YT.PlayerState.PLAYING) player.pauseVideo();
  else player.playVideo();
}

function seekRel(delta) {
  if (!player || !player.getCurrentTime) return;
  player.seekTo(player.getCurrentTime() + delta, true);
}

let muted = false;
function toggleMute() {
  if (!player) return;
  if (muted) { player.unMute(); muted = false; }
  else { player.mute(); muted = true; }
  document.getElementById('mute-btn').textContent = muted ? '🔇' : '🔊';
}

// Bulk-approve state: when [ and ] are pressed, store a custom range,
// then `e` saves it and advances. `b` is whole-video shortcut.
let quickStart = null, quickEnd = null;

function quickApproveWhole() {
  if (!playerReady || !player || !player.getDuration) return;
  const dur = player.getDuration();
  if (!dur || dur <= 0) return;  // duration not yet available, ignore
  ranges = [{start: 0, end: dur, label: 'idle'}];
  currentMode = null;
  saveAndNext();
}

function appendQuickRange() {
  // ] press: commit a range from quickStart..currentTime to ranges[]
  if (!playerReady || !player || !player.getCurrentTime) return false;
  if (quickStart == null) return false;
  const t = player.getCurrentTime();
  if (t <= quickStart + 0.05) {
    console.warn('range too short, ignoring');
    return false;
  }
  ranges.push({start: quickStart, end: t, label: 'idle'});
  quickStart = null; quickEnd = null;
  updateQuickStatus();
  renderRanges();
  return true;
}

function saveQuickRanges() {
  // e press: save accumulated ranges (and any open one) and advance
  if (!playerReady || !player || !player.getDuration) return;
  // If there's still an open quickStart, close it at current time
  if (quickStart != null) {
    appendQuickRange();
  }
  if (ranges.length === 0) {
    console.warn('no ranges to save');
    return;
  }
  currentMode = null;
  saveAndNext();
  quickStart = null; quickEnd = null;
  updateQuickStatus();
}

function updateQuickStatus() {
  const el = document.getElementById('quick-status');
  if (!el) return;
  const s = quickStart != null ? quickStart.toFixed(1) + 's' : '—';
  const e = quickEnd   != null ? quickEnd.toFixed(1)   + 's' : '—';
  el.textContent = 'range: ' + s + ' / ' + e;
}

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  // Drop everything except S (skip without save) while a video is loading.
  // Otherwise stale duration / player state causes silent bugs.
  if (!playerReady && e.key !== 'S') return;
  const k = e.key;
  // Quick approve — use e.code for brackets so layout/shift doesn't matter
  if (k === 'b') { quickApproveWhole(); e.preventDefault(); }
  else if (e.code === 'BracketLeft' || k === '[') {
    if (!player || !player.getCurrentTime) return;
    quickStart = player.getCurrentTime();
    updateQuickStatus();
    e.preventDefault();
  }
  else if (e.code === 'BracketRight' || k === ']') {
    if (!player || !player.getCurrentTime) return;
    appendQuickRange();
    e.preventDefault();
  }
  else if (k === 'e') { saveQuickRanges(); e.preventDefault(); }
  // Per-state label keys
  else if (k === 'i') { setMode('idle'); e.preventDefault(); }
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
  // Transport
  else if (k === ' ') { togglePlay(); e.preventDefault(); }
  else if (k === 'j') { seekRel(-10); e.preventDefault(); }
  else if (k === 'l') { seekRel(10); e.preventDefault(); }
  else if (k === 'ArrowLeft') { seekRel(-5); e.preventDefault(); }
  else if (k === 'ArrowRight') { seekRel(5); e.preventDefault(); }
  else if (k === 'm') { toggleMute(); e.preventDefault(); }
});

// Scrubber
const scrubber = document.getElementById('scrubber');
let scrubbing = false;
scrubber.addEventListener('input', () => { scrubbing = true; });
scrubber.addEventListener('change', () => {
  if (!player || !player.getDuration) return;
  const d = player.getDuration();
  player.seekTo((scrubber.value / 1000) * d, true);
  scrubbing = false;
});

// 200ms tick for time display + play button + scrubber
setInterval(() => {
  if (player && player.getCurrentTime && player.getDuration) {
    const t = player.getCurrentTime();
    const d = player.getDuration();
    document.getElementById('time').textContent = t.toFixed(1) + ' / ' + d.toFixed(1) + 's';
    if (!scrubbing && d > 0) scrubber.value = (t / d) * 1000;
    if (player.getPlayerState) {
      const playing = player.getPlayerState() === YT.PlayerState.PLAYING;
      document.getElementById('play-btn').textContent = playing ? '⏸' : '▶';
    }
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
