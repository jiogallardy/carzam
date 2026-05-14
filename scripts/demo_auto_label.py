"""End-to-end auto-label demo: pick a YouTube video of an untrained car,
run the pipeline, and serve an HTML inspection page.

Defaults:
  * target car  = lexus_lfa (not in the trained class set)
  * video URL   = the "Lexus LFA PURE SOUND" short clip (configurable)
  * references  = auto-scraped from Wikimedia Commons (CC-licensed)

Outputs:
  runs/demo_auto_label/<ts>/
    annotated/*.jpg    # sampled frames with YOLO boxes + DINO match overlay
    report.html        # inspection page
    <video_id>.mp4     # local copy of the YouTube video

Then starts a local HTTP server on the run dir and prints the URL.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import io
import json
import shutil
import socketserver
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from carai.audio import load_wav, resample_to
from carai.data.auto_label import auto_label_video
from carai.data.download import video_id_from_url
from carai.data.screen import (
    CLAP_SR,
    VAD_SR,
    load_clap,
    load_silero_vad,
    screen_window,
)
from carai.data.visual import (
    build_reference_index,
    classify_frame,
    detect_cars,
    load_car_detector,
    probe_video_duration,
    sample_frames_uniform_with_ffmpeg,
)

console = Console()

UA = {"User-Agent": "carAI-demo/0.1 (research; contact: localhost)"}


# ---------- Wikimedia Commons scraper ----------

def search_wikimedia_files(query: str, limit: int = 30) -> list[str]:
    """Return Commons File: titles matching `query`."""
    r = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query", "format": "json", "list": "search",
            "srsearch": query, "srnamespace": "6", "srlimit": str(limit),
        },
        headers=UA, timeout=20,
    )
    r.raise_for_status()
    return [hit["title"] for hit in r.json()["query"]["search"]]


def resolve_file_urls(titles: list[str], thumb_width: int = 1024) -> list[str]:
    """Get thumbnail URLs for File: titles."""
    urls: list[str] = []
    for i in range(0, len(titles), 30):
        batch = "|".join(titles[i:i + 30])
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "titles": batch,
                "prop": "imageinfo", "iiprop": "url|mime",
                "iiurlwidth": str(thumb_width),
            },
            headers=UA, timeout=20,
        )
        r.raise_for_status()
        for p in r.json()["query"]["pages"].values():
            for ii in p.get("imageinfo", []):
                mime = ii.get("mime", "")
                if "image" not in mime or "svg" in mime:
                    continue
                url = ii.get("thumburl") or ii.get("url")
                if url:
                    urls.append(url)
    return urls


def download_images(urls: list[str], out_dir: Path, max_n: int = 15) -> int:
    """Download up to max_n images into out_dir. Returns count saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, url in enumerate(urls):
        if saved >= max_n:
            break
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            if img.width < 200 or img.height < 200:
                continue
            ext = ".jpg"
            img.save(out_dir / f"{saved:02d}{ext}", quality=85)
            saved += 1
        except Exception as e:
            console.print(f"  [yellow]skip[/yellow] {url}: {e}")
    return saved


def ensure_references(refs_dir: Path) -> None:
    """Populate data/references/lexus_lfa/ and data/references/_interior/ if empty."""
    lfa_dir = refs_dir / "lexus_lfa"
    interior_dir = refs_dir / "_interior"

    if not lfa_dir.exists() or len(list(lfa_dir.glob("*.jpg"))) < 5:
        console.print("[bold]scraping Lexus LFA reference photos[/bold]")
        # Multiple queries to maximize coverage of angles/colors
        titles: list[str] = []
        for q in ["Lexus LFA", "Lexus LFA Nurburgring", "Lexus LFA car"]:
            titles.extend(search_wikimedia_files(q, limit=30))
        # De-dup while preserving order
        seen: set[str] = set()
        titles = [t for t in titles if not (t in seen or seen.add(t))]
        urls = resolve_file_urls(titles)
        n = download_images(urls, lfa_dir, max_n=20)
        console.print(f"  saved {n} LFA photos -> {lfa_dir}")

    if not interior_dir.exists() or len(list(interior_dir.glob("*.jpg"))) < 5:
        console.print("[bold]scraping interior reference photos[/bold]")
        titles = []
        for q in [
            "car steering wheel interior",
            "car dashboard cockpit",
            "sports car interior driver POV",
        ]:
            titles.extend(search_wikimedia_files(q, limit=30))
        seen = set()
        titles = [t for t in titles if not (t in seen or seen.add(t))]
        urls = resolve_file_urls(titles)
        n = download_images(urls, interior_dir, max_n=15)
        console.print(f"  saved {n} interior photos -> {interior_dir}")


# ---------- annotate frames with YOLO + DINO ----------

def draw_annotations(
    image: Image.Image,
    dets,
    matches: list[tuple[str, float]],
    target: str,
    verdict_text: str | None = None,
) -> Image.Image:
    """Overlay YOLO boxes + DINOv2 best-match labels on a frame."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    for det, (best_car, sim) in zip(dets, matches):
        x1, y1, x2, y2 = det.bbox
        color = (40, 220, 80) if best_car == target else (220, 60, 60)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = f"{best_car} ({sim:.2f})"
        # background pill
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 4
        draw.rectangle(
            (x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1),
            fill=color,
        )
        draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill="white", font=font)

    if verdict_text:
        # Top-left banner
        tw, th = draw.textbbox((0, 0), verdict_text, font=font_sm)[2:]
        draw.rectangle((0, 0, tw + 16, th + 12), fill=(0, 0, 0, 180))
        draw.text((8, 6), verdict_text, fill="white", font=font_sm)
    return img


# ---------- HTML report ----------

REPORT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>auto-label demo — {car}</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #0e0e10; color: #f4f4f5; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 32px 0 12px; color: #c9c9d1; border-bottom: 1px solid #2c2c30; padding-bottom: 6px; }}
  .meta {{ color: #9a9aa3; margin-bottom: 24px; font-size: 14px; }}
  .video {{ max-width: 720px; aspect-ratio: 16/9; }}
  .video iframe {{ width: 100%; height: 100%; border: 0; border-radius: 8px; }}
  .pre {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
          gap: 12px; }}
  .pre .cell img {{ width: 100%; border-radius: 6px; display: block; }}
  .pre .cell .cap {{ font-size: 12px; color: #b6b6bf; padding: 6px 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #1f1f23; }}
  th {{ background: #15151a; color: #c9c9d1; }}
  .ok {{ color: #4ade80; font-weight: 600; }}
  .no {{ color: #fb7185; font-weight: 600; }}
  .dim {{ color: #71717a; }}
  .timeline {{ position: relative; height: 36px; background: #19191f; border-radius: 6px;
               overflow: hidden; margin: 8px 0 16px; }}
  .timeline .keep {{ position: absolute; top: 0; bottom: 0; background: rgba(74,222,128,0.55); }}
  .timeline .skip {{ position: absolute; top: 0; bottom: 0; background: rgba(120,120,120,0.15); }}
  .scores {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px;
            background: #2a2a31; color: #d4d4d8; margin-right: 4px; }}
  .badge.target {{ background: #14532d; color: #bbf7d0; }}
  .badge.other  {{ background: #7f1d1d; color: #fecaca; }}
  .badge.interior {{ background: #155e75; color: #a5f3fc; }}
  .badge.unrelated {{ background: #3f3f46; color: #a1a1aa; }}
</style></head>
<body><div class="wrap">

<h1>auto-label demo — {car}</h1>
<div class="meta">
  source: <a href="{video_url}" style="color:#a5b4fc">{video_url}</a><br>
  duration: {duration:.1f}s · windows: {n_windows} · kept: {n_kept}
  ({kept_pct:.1f}%) · prefilter: {prefilter_keep}
</div>

<div class="video"><iframe src="https://www.youtube.com/embed/{video_id}"
  allowfullscreen></iframe></div>

<h2>video-level prefilter ({n_prefilter} frames)</h2>
<p class="meta">{prefilter_summary}</p>
<div class="pre">
{prefilter_cells}
</div>

<h2>per-window timeline</h2>
<div class="timeline">{timeline_bars}</div>

<h2>per-window decisions</h2>
<table>
  <thead><tr>
    <th>start</th><th>audio</th><th>visual</th>
    <th>engine</th><th>music</th><th>voice</th><th>speech%</th>
    <th>reasons</th><th>keep</th>
  </tr></thead>
  <tbody>
{window_rows}
  </tbody>
</table>

<h2>per-window annotated frames (kept-only sample)</h2>
<div class="pre">
{kept_frame_cells}
</div>

</div></body></html>
"""


def render_prefilter_cell(rel_path: str, verdict, timestamp: float) -> str:
    if verdict.is_target:
        badge = '<span class="badge target">target</span>'
    elif verdict.is_empty and verdict.is_interior:
        badge = '<span class="badge interior">interior</span>'
    elif verdict.is_empty:
        badge = '<span class="badge unrelated">unrelated</span>'
    elif verdict.other_car_dominant:
        badge = '<span class="badge other">other</span>'
    else:
        badge = '<span class="badge">weak</span>'
    extra = ""
    if verdict.best_match_car:
        extra = f" · best={verdict.best_match_car} ({verdict.best_match_score:.2f})"
    elif verdict.is_empty:
        extra = f" · interior={verdict.interior_score:.2f}"
    return (
        f'<div class="cell"><img src="{rel_path}">'
        f'<div class="cap">{badge} t={timestamp:.1f}s · cars={verdict.n_cars_detected}{extra}</div>'
        f"</div>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", default="lexus_lfa",
                    help="Target class name (creates data/references/<car>/ if missing).")
    ap.add_argument("--url", default="https://youtu.be/F0m6dvpK4m8",
                    help="YouTube URL to inspect.")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    refs_dir = Path("data/references")
    raw_dir = Path("data/raw")
    regions_dir = Path("data/regions")
    video_cache = Path("data/video_cache")

    ensure_references(refs_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or Path("runs/demo_auto_label") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.print(f"[bold]device:[/bold] {device}")

    # 1. Build reference index once (we'll reuse it for annotation)
    console.print("[bold]building reference index[/bold]")
    ref_index = build_reference_index(refs_dir, device)
    console.print(f"  indexed {len(ref_index.car_per_embedding)} crops "
                  f"({len(ref_index.cars)} cars, "
                  f"{0 if ref_index.interior_embeddings is None else ref_index.interior_embeddings.shape[0]} interior refs)")

    # 2. Run the pipeline (downloads video + audio, screens everything)
    console.print(f"\n[bold cyan]running auto_label_video[/bold cyan]  url={args.url} car={args.car}")
    result = auto_label_video(
        url=args.url, car=args.car,
        raw_dir=raw_dir, regions_dir=regions_dir,
        refs_dir=refs_dir, video_cache_dir=video_cache,
        device=device, ref_index=ref_index,
        keep_video_file=True,   # we need the mp4 for re-rendering frames
    )

    video_id = result.video_id
    video_path = video_cache / args.car / f"{video_id}.mp4"
    if not video_path.exists():
        console.print(f"[red]video file missing[/red] {video_path}")
        return

    # 3. Re-sample the same prefilter timestamps and draw annotations
    console.print("\n[bold]annotating prefilter frames[/bold]")
    detector = load_car_detector()
    prefilter = result.video_prefilter or {}
    pre_ts = prefilter.get("timestamps", [])
    if not pre_ts:
        pre_ts, _ = sample_frames_uniform_with_ffmpeg(video_path, 10)
    pre_frames = sample_frames_uniform_with_ffmpeg(video_path, len(pre_ts))[1]

    prefilter_cells: list[str] = []
    for i, (ts_i, frame) in enumerate(zip(pre_ts, pre_frames)):
        dets = detect_cars(frame, detector)
        from carai.data.visual import classify_crops, crop_box
        if dets:
            crops = [crop_box(frame, d.bbox) for d in dets]
            matches = classify_crops(crops, ref_index, device)
        else:
            matches = []
        # Re-derive verdict for label consistency
        verdict = classify_frame(frame, args.car, detector, ref_index, device)
        # Build a header banner showing the conclusion
        if verdict.is_target:
            banner = f"TARGET ({args.car}) {verdict.best_match_score:.2f}"
        elif verdict.is_empty and verdict.is_interior:
            banner = f"interior  sim={verdict.interior_score:.2f}"
        elif verdict.is_empty:
            banner = f"empty  interior_sim={verdict.interior_score:.2f}"
        elif verdict.other_car_dominant:
            banner = f"OTHER ({verdict.best_match_car}) {verdict.best_match_score:.2f}"
        else:
            banner = f"weak  {verdict.best_match_car} {verdict.best_match_score:.2f}"
        annotated = draw_annotations(frame, dets, matches, args.car, banner)
        rel = f"annotated/pre_{i:02d}.jpg"
        annotated.save(out_dir / rel, quality=85)
        prefilter_cells.append(render_prefilter_cell(rel, verdict, ts_i))

    # 4. Also annotate a few of the kept windows so the user can see what
    #    "passed" frames look like.
    console.print("[bold]annotating sample of kept windows[/bold]")
    kept = [d for d in result.decisions if d.keep]
    kept_to_show = kept[:8]
    kept_cells: list[str] = []
    for j, dec in enumerate(kept_to_show):
        mid_ts = (dec.start + dec.end) / 2
        frames = sample_frames_uniform_with_ffmpeg(video_path, 1, duration=mid_ts * 2 + 0.1)[1] if mid_ts > 0 else []
        # Better: pull a single frame at mid_ts
        from carai.data.visual import sample_frames_with_ffmpeg
        single = sample_frames_with_ffmpeg(video_path, [mid_ts])
        if not single:
            continue
        frame = single[0]
        dets = detect_cars(frame, detector)
        from carai.data.visual import classify_crops, crop_box
        if dets:
            matches = classify_crops([crop_box(frame, d.bbox) for d in dets], ref_index, device)
        else:
            matches = []
        verdict = classify_frame(frame, args.car, detector, ref_index, device)
        banner = f"window {dec.start:.1f}s–{dec.end:.1f}s · KEPT"
        annotated = draw_annotations(frame, dets, matches, args.car, banner)
        rel = f"annotated/kept_{j:02d}.jpg"
        annotated.save(out_dir / rel, quality=85)
        kept_cells.append(render_prefilter_cell(rel, verdict, mid_ts))

    # 5. Build timeline bars
    bars: list[str] = []
    for d in result.decisions:
        left_pct = 100.0 * d.start / max(0.001, result.duration)
        width_pct = 100.0 * (d.end - d.start) / max(0.001, result.duration)
        cls = "keep" if d.keep else "skip"
        bars.append(
            f'<div class="{cls}" style="left:{left_pct:.2f}%;width:{width_pct:.2f}%" '
            f'title="{d.start:.1f}–{d.end:.1f} {"KEEP" if d.keep else "skip"}"></div>'
        )

    # 6. Per-window table rows
    rows: list[str] = []
    for d in result.decisions:
        eng = d.clap.get("engine", 0.0)
        mus = d.clap.get("music", 0.0)
        voc = d.clap.get("voice", 0.0)
        a_ok = '<span class="ok">ok</span>' if d.audio_clean else '<span class="no">no</span>'
        v_ok = '<span class="ok">ok</span>' if d.visual_ok else '<span class="no">no</span>'
        keep = '<span class="ok">KEEP</span>' if d.keep else '<span class="dim">skip</span>'
        reasons = "; ".join(d.audio_reasons + ([d.visual_reason] if not d.visual_ok else []))
        rows.append(
            f"<tr><td>{d.start:5.1f}s</td><td>{a_ok}</td><td>{v_ok}</td>"
            f"<td class='scores'>{eng:.2f}</td><td class='scores'>{mus:.2f}</td>"
            f"<td class='scores'>{voc:.2f}</td><td class='scores'>—</td>"
            f"<td class='dim'>{reasons}</td><td>{keep}</td></tr>"
        )

    # 7. Summary
    n_kept = sum(1 for d in result.decisions if d.keep)
    kept_pct = 100.0 * n_kept / max(1, len(result.decisions))
    pre_summary = (
        f"target={prefilter.get('n_target',0)}, "
        f"interior={prefilter.get('n_interior',0)}, "
        f"other={prefilter.get('n_other',0)}, "
        f"unrelated={prefilter.get('n_empty_unrelated',0)} → "
        f"acceptable_ratio={prefilter.get('acceptable_ratio',0):.2f}"
    )

    html = REPORT_HTML.format(
        car=args.car,
        video_url=args.url,
        video_id=video_id,
        duration=result.duration,
        n_windows=len(result.decisions),
        n_kept=n_kept,
        kept_pct=kept_pct,
        prefilter_keep="KEEP" if result.video_kept else "DROPPED",
        n_prefilter=len(pre_ts),
        prefilter_summary=pre_summary,
        prefilter_cells="\n".join(prefilter_cells),
        timeline_bars="\n".join(bars),
        window_rows="\n".join(rows),
        kept_frame_cells="\n".join(kept_cells) if kept_cells else "<div class='meta'>no kept windows</div>",
    )
    (out_dir / "report.html").write_text(html)
    console.print(f"\n[green]wrote[/green] {out_dir / 'report.html'}")

    # 8. Serve
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
