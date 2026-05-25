"""Serve a browser report of every source video used for one class.

For each YouTube URL ingested for the target class, embed it via iframe with
clickable buttons that seek to each auto-label-accepted timestamp range.
Lets you spot-check that the audio gate actually kept slices of the
intended car and not unrelated footage.

Usage:
    .venv/bin/python scripts/verify_class_videos.py --car porsche_carrera_gt
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import socketserver
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console

from carzam.data.download import video_id_from_url

console = Console()


def load_sources_for_car(car: str, files: list[Path]) -> list[str]:
    """Walk each sources YAML, collect URLs listed under this car."""
    urls: list[str] = []
    seen: set[str] = set()
    for f in files:
        if not f.exists():
            continue
        data = yaml.safe_load(f.read_text()) or {}
        for u in data.get(car, []) or []:
            if u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def load_accept_regions(yaml_path: Path) -> tuple[list[tuple[float, float]], float]:
    """Return (accept_ranges, total_duration). Filters out 'skip' label."""
    if not yaml_path.exists():
        return [], 0.0
    data = yaml.safe_load(yaml_path.read_text()) or {}
    duration = float(data.get("duration") or 0.0)
    ranges = []
    for r in data.get("ranges") or []:
        if r.get("label") == "skip":
            continue
        ranges.append((float(r["start"]), float(r["end"])))
    return ranges, duration


def fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


REPORT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>verify {car} videos</title>
<style>
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,sans-serif;
          background:#0e0e10; color:#f4f4f5; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:24px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:24px 0 8px; color:#c9c9d1; }}
  .meta {{ color:#9a9aa3; margin-bottom:24px; font-size:14px; }}
  .video-card {{ background:#15151a; border-radius:10px; padding:16px;
                 margin-bottom:24px; }}
  .video-card h3 {{ margin:0 0 4px; font-size:15px; }}
  .video-card .url {{ font-size:12px; color:#9a9aa3; word-break:break-all; }}
  .player-wrap {{ position:relative; width:100%; aspect-ratio:16/9;
                  margin:10px 0; background:#000; border-radius:6px; overflow:hidden; }}
  .player-wrap iframe {{ width:100%; height:100%; border:0; }}
  .stats {{ font-size:13px; color:#c9c9d1; margin:6px 0; }}
  .stat-pill {{ display:inline-block; background:#1f1f23; padding:2px 8px;
                border-radius:4px; margin-right:6px; font-family:ui-monospace,Menlo,monospace; }}
  .ranges {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
  .range-btn {{ background:#14532d; color:#bbf7d0; border:0; padding:6px 10px;
                border-radius:4px; cursor:pointer; font-family:ui-monospace,Menlo,monospace;
                font-size:12px; }}
  .range-btn:hover {{ background:#166534; }}
  .range-btn.full {{ background:#1e3a8a; color:#bfdbfe; }}
  .range-btn.full:hover {{ background:#1d4ed8; }}
  .dim {{ color:#71717a; }}
</style></head>
<body><div class="wrap">

<h1>verify videos — {car}</h1>
<div class="meta">
Per-video accept ranges kept by the auto-label audio gate. Click a green
button to play that exact slice; click <code>full</code> to watch the
whole video. Use this to confirm the kept regions actually contain the
intended car.<br>
<span class="stat-pill">videos: {n_videos}</span>
<span class="stat-pill">total accepted: {total_accepted:.0f}s</span>
<span class="stat-pill">total duration: {total_duration:.0f}s</span>
<span class="stat-pill">keep rate: {keep_rate:.1f}%</span>
</div>

{cards}

</div></body></html>
"""

CARD_HTML = """
<div class="video-card" id="card-{vid}">
  <h3>{vid}</h3>
  <div class="url"><a href="{url}" style="color:#a5b4fc">{url}</a></div>
  <div class="player-wrap"><iframe id="iframe-{vid}"
    src="https://www.youtube.com/embed/{vid}"
    allow="autoplay; encrypted-media" allowfullscreen></iframe></div>
  <div class="stats">
    <span class="stat-pill">accepted: {accepted:.0f}s</span>
    <span class="stat-pill">duration: {duration:.0f}s</span>
    <span class="stat-pill">{n_ranges} ranges</span>
  </div>
  <div class="ranges">
    <button class="range-btn full"
      onclick="play('{vid}', null, null)">▶ full</button>
    {range_buttons}
  </div>
</div>
"""

# Range buttons need to seek the iframe. YouTube IFrame Player API requires
# `enablejsapi=1` on the iframe src and a small loader. We hot-swap the
# iframe src instead — simpler, no extra JS dependency.
JS = """
<script>
function play(vid, start, end) {
  const iframe = document.getElementById('iframe-' + vid);
  let src = 'https://www.youtube.com/embed/' + vid + '?autoplay=1';
  if (start !== null) src += '&start=' + Math.floor(start);
  if (end !== null)   src += '&end=' + Math.ceil(end);
  iframe.src = src;
}
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", required=True)
    ap.add_argument("--regions-dir", type=Path, default=Path("data/regions"))
    ap.add_argument("--sources", type=Path, nargs="*", default=[
        Path("config/sources_expanded.yaml"),
        Path("config/sources.yaml"),
    ])
    ap.add_argument("--port", type=int, default=8014)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    urls = load_sources_for_car(args.car, args.sources)
    if not urls:
        console.print(f"[red]no source URLs found for {args.car!r}[/red]")
        console.print(f"  checked: {args.sources}")
        return

    console.print(f"[bold]{args.car}[/bold]  {len(urls)} videos in sources")

    cards: list[str] = []
    total_accepted = 0.0
    total_duration = 0.0
    for url in urls:
        try:
            vid = video_id_from_url(url)
        except ValueError:
            console.print(f"  [yellow]bad url[/yellow] {url}")
            continue
        regions_path = args.regions_dir / args.car / f"{vid}.yaml"
        ranges, duration = load_accept_regions(regions_path)
        accepted = sum(e - s for s, e in ranges)
        total_accepted += accepted
        total_duration += duration

        range_buttons = []
        for s, e in ranges:
            range_buttons.append(
                f'<button class="range-btn" onclick="play(\'{vid}\', {s}, {e})">'
                f'▶ {fmt_seconds(s)}–{fmt_seconds(e)}'
                f'</button>'
            )
        # Cap displayed range buttons at 40 — for very long videos with many
        # short accepts the UI gets noisy.
        if len(range_buttons) > 40:
            displayed = range_buttons[:40]
            displayed.append(
                f'<span class="dim">+{len(range_buttons) - 40} more ranges</span>'
            )
            range_buttons = displayed
        cards.append(CARD_HTML.format(
            vid=vid, url=url, accepted=accepted, duration=duration,
            n_ranges=len(ranges),
            range_buttons="\n".join(range_buttons) or
            '<span class="dim">no accepted ranges (whole video skipped)</span>',
        ))

    keep_rate = 100.0 * total_accepted / max(0.001, total_duration)

    html = REPORT_HTML.format(
        car=args.car,
        n_videos=len(urls),
        total_accepted=total_accepted,
        total_duration=total_duration,
        keep_rate=keep_rate,
        cards="\n".join(cards),
    ) + JS

    out_dir = Path("runs/verify_videos") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(html)
    console.print(f"[green]wrote[/green] {out_dir / 'report.html'}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(out_dir))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/report.html"
        console.print(f"\n[bold cyan]serving on {url}[/bold cyan]  (ctrl-c to stop)")
        if not args.no_open:
            threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)),
                             daemon=True).start()
        with contextlib.suppress(KeyboardInterrupt):
            httpd.serve_forever()


if __name__ == "__main__":
    main()
