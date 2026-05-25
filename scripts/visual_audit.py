"""Post-hoc visual audit of auto-labeled data.

For each car in data/regions/, for every accepted video:
  1. Pull a few thumbnails (maxres/hq/sd) from YouTube as visual refs for
     ALL cars (not just the target). Thumbnails match production distribution
     better than Wikipedia studio photos.
  2. Sample 5-8 frames at the midpoints of accepted regions.
  3. Run YOLO car detection on each frame, then DINOv2 embed each car crop
     and find the nearest-prototype car across the WHOLE reference index.
  4. Flag videos where:
       - The target car appears in fewer than `--min-target-rate` frames, OR
       - Some OTHER car appears in more than `--max-other-rate` frames

Outputs an HTML report at runs/visual_audit/<ts>/report.html with thumbnails
of suspicious videos + their predicted dominant cars, so you can spot-check
and feed back to scripts/block_videos.py for cleanup.

Usage:
    .venv/bin/python scripts/visual_audit.py --car porsche_carrera_gt
    .venv/bin/python scripts/visual_audit.py --all
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import io
import json
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from rich.console import Console

# Re-use the polite scraper helpers
sys.path.insert(0, str(Path(__file__).parent))
from demo_auto_label import _polite_get  # type: ignore[import-not-found]

from carzam.data.download import video_id_from_url
from carzam.data.visual import (
    classify_crops,
    crop_box,
    detect_cars,
    load_car_detector,
    load_image_encoder,
    sample_frames_with_ffmpeg,
)

console = Console()


# ---------- thumbnail refs ----------

YT_THUMB_URLS = [
    "https://i.ytimg.com/vi/{vid}/maxresdefault.jpg",
    "https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
    "https://i.ytimg.com/vi/{vid}/mqdefault.jpg",
]


def fetch_thumbnails_for_videos(video_ids: list[str], out_dir: Path,
                                 max_per_video: int = 2) -> int:
    """Download up to max_per_video thumbnails per video to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for vid in video_ids:
        added_for_video = 0
        for tmpl in YT_THUMB_URLS:
            if added_for_video >= max_per_video:
                break
            url = tmpl.format(vid=vid)
            target = out_dir / f"{vid}_{added_for_video}.jpg"
            if target.exists():
                added_for_video += 1
                saved += 1
                continue
            try:
                r = _polite_get(url, attempts=2)
                if r.status_code != 200 or len(r.content) < 5_000:
                    continue
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                if img.width < 200:
                    continue
                img.save(target, quality=85)
                added_for_video += 1
                saved += 1
            except Exception:
                continue
    return saved


# ---------- per-video frame sampling ----------

def _yt_dlp_cmd() -> list[str]:
    from shutil import which
    binary = which("yt-dlp")
    if binary:
        return [binary]
    return [sys.executable, "-m", "yt_dlp"]


def stream_video_for_frames(
    url: str, vid: str, ts_list: list[float], cache_dir: Path,
) -> list[Image.Image]:
    """Download a small mp4 (cached), then extract frames at the given
    timestamps via ffmpeg. Cheaper than downloading-by-section because
    yt-dlp section downloads need ffmpeg postprocessing anyway."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp4 = cache_dir / f"{vid}.mp4"
    if not mp4.exists():
        cmd = [
            *_yt_dlp_cmd(),
            "-f", "best[height<=360][ext=mp4]/best[ext=mp4]/best",
            "--no-warnings", "--quiet",
            "-o", str(mp4),
            url,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=180)
        except Exception:
            return []
    if not mp4.exists():
        return []
    return sample_frames_with_ffmpeg(mp4, ts_list)


# ---------- build the multi-class ref index ----------

def build_thumbnail_ref_index(
    refs_root: Path, cars: list[str], device: torch.device,
):
    """Build a class-keyed embedding index from per-car thumbnail dirs.
    Each thumbnail is YOLO-cropped (largest car) before embedding."""
    detector = load_car_detector()
    encoder, processor = load_image_encoder()
    encoder = encoder.to(device).eval()

    all_embs: list[torch.Tensor] = []
    all_cars: list[str] = []

    @torch.no_grad()
    def _embed(imgs: list[Image.Image]) -> torch.Tensor:
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = encoder(**inputs)
        emb = out.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(emb, dim=-1)

    for car in cars:
        car_dir = refs_root / car
        if not car_dir.exists():
            continue
        imgs: list[Image.Image] = []
        for p in sorted(car_dir.glob("*.jpg")):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            dets = detect_cars(img, detector, min_confidence=0.20, min_area_ratio=0.005)
            if dets:
                dets.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                          reverse=True)
                imgs.append(crop_box(img, dets[0].bbox))
            else:
                imgs.append(img)  # fall back to whole frame
        if not imgs:
            continue
        # Batch
        for i in range(0, len(imgs), 16):
            batch_embs = _embed(imgs[i:i + 16])
            all_embs.append(batch_embs)
            all_cars.extend([car] * batch_embs.shape[0])

    if not all_embs:
        raise RuntimeError("no thumbnails embedded — refs_root is empty?")

    embeddings = torch.cat(all_embs, dim=0)
    return {
        "embeddings": embeddings,
        "car_per_emb": tuple(all_cars),
        "encoder": encoder,
        "processor": processor,
        "detector": detector,
    }


# ---------- per-video audit ----------

def audit_video(
    car: str, url: str, vid: str, regions_yaml: Path,
    ref_index: dict, mp4_cache: Path, device: torch.device,
    n_frames: int = 6,
) -> dict:
    """Return per-video stats and per-frame predictions."""
    data = yaml.safe_load(regions_yaml.read_text()) or {}
    accept_ranges = [
        (float(r["start"]), float(r["end"]))
        for r in (data.get("ranges") or [])
        if r.get("label") != "skip"
    ]
    if not accept_ranges:
        return {"car": car, "vid": vid, "url": url, "skipped": "no-accept-ranges"}

    # Sample frames at midpoints of accept ranges (spread evenly)
    if len(accept_ranges) <= n_frames:
        ts_list = [(s + e) / 2 for s, e in accept_ranges]
    else:
        step = len(accept_ranges) / n_frames
        ts_list = [
            (accept_ranges[int(i * step)][0] + accept_ranges[int(i * step)][1]) / 2
            for i in range(n_frames)
        ]

    frames = stream_video_for_frames(url, vid, ts_list, mp4_cache)
    if not frames:
        return {"car": car, "vid": vid, "url": url, "skipped": "no-frames"}

    # Classify each frame
    from carzam.data.visual import ReferenceIndex
    ri = ReferenceIndex(
        cars=tuple(sorted(set(ref_index["car_per_emb"]))),
        embeddings=ref_index["embeddings"],
        car_per_embedding=ref_index["car_per_emb"],
        encoder_name="dinov2",
        encoder=ref_index["encoder"],
        processor=ref_index["processor"],
    )

    frame_results = []
    for ts, frame in zip(ts_list, frames):
        dets = detect_cars(frame, ref_index["detector"])
        best_car, best_sim, n_cars = None, 0.0, 0
        if dets:
            crops = [crop_box(frame, d.bbox) for d in dets]
            matches = classify_crops(crops, ri, device)
            best_idx = max(range(len(matches)), key=lambda i: matches[i][1])
            best_car, best_sim = matches[best_idx]
            n_cars = len(dets)
        frame_results.append({
            "ts": ts, "best_car": best_car, "best_sim": float(best_sim),
            "n_cars": n_cars,
        })

    pred_counter: Counter[str] = Counter()
    for r in frame_results:
        if r["best_car"]:
            pred_counter[r["best_car"]] += 1
    top_pred, top_count = (pred_counter.most_common(1) or [(None, 0)])[0]

    n_target = sum(1 for r in frame_results if r["best_car"] == car)
    n_total = len(frame_results)
    target_rate = n_target / max(1, n_total)
    other_dominant = top_pred and top_pred != car and top_count > n_target

    return {
        "car": car, "vid": vid, "url": url,
        "n_frames": n_total, "n_target": n_target,
        "target_rate": target_rate,
        "predicted_dominant": top_pred,
        "predicted_count": top_count,
        "other_dominant": bool(other_dominant),
        "frame_results": frame_results,
    }


# ---------- HTML report ----------

REPORT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>visual audit — {title}</title>
<style>
  body{{margin:0;font-family:-apple-system,sans-serif;background:#0e0e10;color:#f4f4f5;}}
  .wrap{{max-width:1200px;margin:0 auto;padding:24px;}}
  h1{{font-size:22px;margin:0 0 4px;}}
  h2{{font-size:17px;margin:24px 0 8px;color:#c9c9d1;}}
  .summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0;}}
  .stat{{background:#15151a;padding:10px;border-radius:6px;}}
  .stat .v{{font-size:22px;font-family:ui-monospace,Menlo,monospace;font-weight:600;}}
  .stat .k{{font-size:11px;color:#9a9aa3;text-transform:uppercase;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #1f1f23;}}
  th{{background:#15151a;color:#c9c9d1;}}
  .ok{{color:#4ade80;}} .bad{{color:#fb7185;}} .dim{{color:#71717a;}}
  .video-card{{background:#15151a;border-radius:8px;padding:12px;margin-bottom:14px;}}
  .video-card iframe{{width:100%;aspect-ratio:16/9;border:0;border-radius:6px;}}
  .video-card .meta{{font-size:12px;color:#b6b6bf;margin-top:6px;}}
  code{{background:#1f1f23;padding:1px 4px;border-radius:3px;}}
</style></head><body><div class="wrap">

<h1>visual audit — {title}</h1>
<div class="summary">
  <div class="stat"><div class="k">videos audited</div><div class="v">{n_audited}</div></div>
  <div class="stat"><div class="k">flagged</div><div class="v bad">{n_flagged}</div></div>
  <div class="stat"><div class="k">clean</div><div class="v ok">{n_clean}</div></div>
  <div class="stat"><div class="k">skipped (errors)</div><div class="v dim">{n_skipped}</div></div>
</div>

<h2>flagged videos (target not dominant)</h2>
{flagged_section}

<h2>all results</h2>
<table><thead><tr>
<th>car</th><th>video</th><th>frames</th><th>target%</th><th>predicted</th><th>note</th>
</tr></thead><tbody>
{table_rows}
</tbody></table>

<h2>blocklist snippet (paste into next cleanup)</h2>
<pre>{blocklist}</pre>

</div></body></html>
"""


def render_flagged_card(r: dict) -> str:
    pct = 100.0 * r["target_rate"]
    return f"""
<div class="video-card">
  <iframe src="https://www.youtube.com/embed/{r["vid"]}"
    allow="autoplay" allowfullscreen></iframe>
  <div class="meta">
    <strong>{r["car"]}</strong> · <code>{r["vid"]}</code> · target {pct:.0f}% · predicted
    dominant <span class="bad">{r["predicted_dominant"]}</span> ({r["predicted_count"]}/{r["n_frames"]})
  </div>
</div>
"""


def render_table_row(r: dict) -> str:
    if r.get("skipped"):
        return (
            f"<tr><td>{r.get('car','')}</td><td><code>{r.get('vid','')}</code></td>"
            f"<td class='dim'>—</td><td class='dim'>—</td>"
            f"<td class='dim'>—</td><td class='dim'>{r['skipped']}</td></tr>"
        )
    pct = 100.0 * r["target_rate"]
    pred_class = "bad" if r["other_dominant"] else "dim"
    target_class = "bad" if r["target_rate"] < 0.3 else "ok"
    return (
        f"<tr><td>{r['car']}</td><td><code>{r['vid']}</code></td>"
        f"<td>{r['n_target']}/{r['n_frames']}</td>"
        f"<td class='{target_class}'>{pct:.0f}%</td>"
        f"<td class='{pred_class}'>{r['predicted_dominant'] or '—'}</td>"
        f"<td class='dim'>{'flagged' if r['other_dominant'] or r['target_rate'] < 0.3 else 'ok'}</td></tr>"
    )


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", help="One car to audit; or use --all.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cars", help="Comma-separated subset of cars.")
    ap.add_argument("--regions-dir", type=Path, default=Path("data/regions"))
    ap.add_argument("--sources", type=Path, nargs="*", default=[
        Path("config/sources_expanded.yaml"),
        Path("config/sources.yaml"),
    ])
    ap.add_argument("--thumb-cache", type=Path,
                    default=Path("data/visual_audit_cache/thumbnails"))
    ap.add_argument("--mp4-cache", type=Path,
                    default=Path("data/visual_audit_cache/mp4s"))
    ap.add_argument("--n-frames-per-video", type=int, default=6)
    ap.add_argument("--port", type=int, default=8015)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    # Pick cars to audit.
    if args.all:
        cars_to_audit = sorted(
            d.name for d in args.regions_dir.iterdir() if d.is_dir()
        )
    elif args.cars:
        cars_to_audit = [c.strip() for c in args.cars.split(",")]
    elif args.car:
        cars_to_audit = [args.car]
    else:
        ap.error("specify --car, --cars, or --all")

    # Load all sources.
    sources: dict[str, list[str]] = {}
    for sf in args.sources:
        if sf.exists():
            data = yaml.safe_load(sf.read_text()) or {}
            for k, v in data.items():
                sources.setdefault(k, [])
                for u in (v or []):
                    if u not in sources[k]:
                        sources[k].append(u)

    # Build ref index — but we need refs for EVERY car so we can detect
    # "wrong car visible". Fetch thumbnails for all cars that have sources.
    all_cars = sorted(sources.keys())
    console.print(f"[bold]fetching thumbnails[/bold] for {len(all_cars)} cars...")
    for c in all_cars:
        vids = []
        for u in sources.get(c, []):
            try:
                vids.append(video_id_from_url(u))
            except ValueError:
                continue
        n = fetch_thumbnails_for_videos(vids, args.thumb_cache / c)
        if n:
            console.print(f"  {c}: {n} thumbs")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.print("[bold]building ref index[/bold]")
    ref_index = build_thumbnail_ref_index(args.thumb_cache, all_cars, device)
    console.print(f"  indexed {ref_index['embeddings'].shape[0]} thumbnails "
                  f"across {len(set(ref_index['car_per_emb']))} cars")

    # Audit each car
    results: list[dict] = []
    for car in cars_to_audit:
        regions_dir = args.regions_dir / car
        if not regions_dir.exists():
            continue
        url_by_vid: dict[str, str] = {}
        for u in sources.get(car, []):
            try:
                url_by_vid[video_id_from_url(u)] = u
            except ValueError:
                pass
        for yaml_path in sorted(regions_dir.glob("*.yaml")):
            vid = yaml_path.stem
            url = url_by_vid.get(vid, f"https://youtu.be/{vid}")
            console.print(f"  auditing {car}/{vid}")
            r = audit_video(car, url, vid, yaml_path, ref_index,
                            args.mp4_cache, device, args.n_frames_per_video)
            results.append(r)

    # Render report
    flagged = [r for r in results
               if not r.get("skipped") and (r.get("other_dominant") or r.get("target_rate", 1.0) < 0.3)]
    clean = [r for r in results
             if not r.get("skipped") and r not in flagged]
    skipped = [r for r in results if r.get("skipped")]

    blocklist_lines = []
    for r in flagged:
        blocklist_lines.append(f"{r['car']}/{r['vid']}  # pred={r['predicted_dominant']}, target={100*r['target_rate']:.0f}%")
    blocklist = "\n".join(blocklist_lines) or "(no videos flagged)"

    html = REPORT_HTML.format(
        title=", ".join(cars_to_audit) if len(cars_to_audit) <= 5
               else f"{len(cars_to_audit)} cars",
        n_audited=len(results), n_flagged=len(flagged),
        n_clean=len(clean), n_skipped=len(skipped),
        flagged_section="\n".join(render_flagged_card(r) for r in flagged)
                        or "<div class='dim'>nothing flagged</div>",
        table_rows="\n".join(render_table_row(r) for r in results),
        blocklist=blocklist,
    )

    out_dir = Path("runs/visual_audit") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(html)
    (out_dir / "summary.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "frame_results"}
                    for r in results], indent=2)
    )
    console.print(f"\n[green]wrote[/green] {out_dir / 'report.html'}")
    console.print(f"  {len(flagged)} flagged, {len(clean)} clean, {len(skipped)} skipped")

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
