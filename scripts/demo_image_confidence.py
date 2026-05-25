"""Discrimination test for the visual pipeline.

Scrapes fresh Lexus LFA photos (different Wikimedia queries than the
reference library, so the test set is NOT what we indexed) plus a handful
of negative-control cars, runs each through YOLO + DINOv2 against the
existing reference index, and renders a sortable HTML report.

If the visual gate works, fresh LFA shots score *higher* against the
lexus_lfa reference embeddings than the negative controls do — that
gap is what would let us discriminate make/model at scale.

Usage:
    .venv/bin/python scripts/demo_image_confidence.py --port 8012
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import io
import socketserver
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from rich.console import Console

# Reuse the scraper from the other demo so we hit Wikimedia politely.
from demo_auto_label import (  # type: ignore[import-not-found]
    _polite_get,
    download_images,
    draw_annotations,
    resolve_file_urls,
    search_wikimedia_files,
)

from carzam.data.visual import (
    build_reference_index,
    classify_crops,
    classify_frame,
    crop_box,
    detect_cars,
    load_car_detector,
    score_interior,
)

console = Console()


# Test set definition. Each entry: (label-shown, expected-positive, queries)
# "expected_positive" = should score HIGH against the lexus_lfa refs.
TEST_BUCKETS: list[tuple[str, bool, list[str]]] = [
    ("Lexus LFA (fresh)",         True,  ["Lexus LFA Spider", "Lexus LFA roadster"]),
    ("Ferrari 458",               False, ["Ferrari 458", "Ferrari 458 Italia"]),
    ("Lamborghini Huracan",       False, ["Lamborghini Huracan"]),
    ("Porsche 911 GT3",           False, ["Porsche 911 GT3", "Porsche GT3"]),
    ("McLaren 720S",              False, ["McLaren 720S"]),
    ("Audi R8",                   False, ["Audi R8 V10"]),
    ("Toyota Camry (generic)",    False, ["Toyota Camry"]),
]


def fetch_bucket(queries: list[str], target_count: int, out_dir: Path) -> list[Path]:
    """Scrape `target_count` images for this bucket if not already cached."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(out_dir.glob("*.jpg"))
    if len(cached) >= target_count:
        return cached[:target_count]

    titles: list[str] = []
    for q in queries:
        try:
            titles.extend(search_wikimedia_files(q, limit=20))
        except Exception as e:
            console.print(f"  [yellow]search failed[/yellow] {q}: {e}")
    # de-dup preserving order
    seen: set[str] = set()
    titles = [t for t in titles if not (t in seen or seen.add(t))]
    try:
        urls = resolve_file_urls(titles)
    except Exception as e:
        console.print(f"  [yellow]url resolve failed[/yellow]: {e}")
        urls = []
    n = download_images(urls, out_dir, max_n=target_count + 2)
    console.print(f"  saved {n} -> {out_dir}")
    return sorted(out_dir.glob("*.jpg"))[:target_count]


REPORT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>visual confidence test — lexus_lfa</title>
<style>
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,sans-serif;
          background:#0e0e10; color:#f4f4f5; }}
  .wrap {{ max-width:1280px; margin:0 auto; padding:24px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:28px 0 10px; color:#c9c9d1;
        border-bottom:1px solid #2c2c30; padding-bottom:6px; }}
  .meta {{ color:#9a9aa3; margin-bottom:24px; font-size:14px; line-height:1.5; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr));
           gap:12px; }}
  .cell {{ background:#15151a; border-radius:8px; padding:8px; }}
  .cell img {{ width:100%; border-radius:6px; display:block; }}
  .cap {{ font-size:12px; color:#b6b6bf; padding:8px 4px 2px; line-height:1.4; }}
  .score {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:14px;
            font-weight:600; }}
  .high {{ color:#4ade80; }}
  .mid  {{ color:#fbbf24; }}
  .low  {{ color:#fb7185; }}
  .badge {{ display:inline-block; padding:1px 6px; border-radius:3px;
            font-size:11px; background:#2a2a31; color:#d4d4d8; margin-right:4px; }}
  .badge.pos {{ background:#14532d; color:#bbf7d0; }}
  .badge.neg {{ background:#7f1d1d; color:#fecaca; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:16px; }}
  th, td {{ padding:6px 10px; text-align:left; border-bottom:1px solid #1f1f23; }}
  th {{ background:#15151a; color:#c9c9d1; }}
  .summary {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;
              margin:16px 0 24px; }}
  .stat {{ background:#15151a; border-radius:8px; padding:14px; }}
  .stat .v {{ font-size:24px; font-weight:600; font-family:ui-monospace,Menlo,monospace; }}
  .stat .k {{ font-size:12px; color:#9a9aa3; text-transform:uppercase;
              letter-spacing:.05em; }}
</style></head>
<body><div class="wrap">

<h1>visual confidence test — lexus_lfa</h1>
<div class="meta">
Each image is run through YOLOv8 (car detection) + DINOv2 (cosine sim
against the curated <code>data/references/lexus_lfa/</code> set, 18 photos).<br>
The score shown is the cosine similarity of the strongest detected car
crop against the closest LFA reference. <b>If the pipeline is working,
the green (Lexus LFA) bucket should sit higher than the rest.</b>
</div>

<div class="summary">
  <div class="stat"><div class="k">LFA mean</div><div class="v high">{lfa_mean:.3f}</div></div>
  <div class="stat"><div class="k">non-LFA mean</div><div class="v low">{neg_mean:.3f}</div></div>
  <div class="stat"><div class="k">separation</div><div class="v">{sep:.3f}</div></div>
</div>

<h2>per-bucket summary</h2>
<table><thead><tr>
<th>bucket</th><th>n</th><th>mean sim</th><th>max sim</th><th>min sim</th>
<th>above thresh (0.55)</th>
</tr></thead><tbody>
{bucket_rows}
</tbody></table>

<h2>all images, ranked by sim to lexus_lfa refs (high → low)</h2>
<div class="grid">
{cells}
</div>

</div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--n-per-bucket", type=int, default=6)
    ap.add_argument("--target-threshold", type=float, default=0.55,
                    help="Match threshold used by classify_frame.")
    args = ap.parse_args()

    refs_dir = Path("data/references")
    if not (refs_dir / "lexus_lfa").exists():
        raise SystemExit(
            "data/references/lexus_lfa/ is empty. Run "
            "scripts/demo_auto_label.py first to populate references."
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/demo_confidence") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    cache_dir = Path("data/confidence_test_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.print(f"[bold]device:[/bold] {device}")

    console.print("[bold]building reference index[/bold]")
    ref_index = build_reference_index(refs_dir, device)
    console.print(f"  indexed {len(ref_index.car_per_embedding)} crops "
                  f"({len(ref_index.cars)} cars, "
                  f"{0 if ref_index.interior_embeddings is None else ref_index.interior_embeddings.shape[0]} interior)")
    detector = load_car_detector()

    # ----- fetch test images per bucket -----
    bucket_images: dict[str, tuple[bool, list[Path]]] = {}
    for label, is_positive, queries in TEST_BUCKETS:
        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        console.print(f"[bold]fetching[/bold] {label}")
        paths = fetch_bucket(queries, args.n_per_bucket, cache_dir / slug)
        bucket_images[label] = (is_positive, paths)

    # ----- score each image -----
    rows: list[dict] = []   # for sorting + rendering
    for label, (is_positive, paths) in bucket_images.items():
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            dets = detect_cars(img, detector)
            if dets:
                crops = [crop_box(img, d.bbox) for d in dets]
                matches = classify_crops(crops, ref_index, device)
                # max sim across detected cars
                best_idx = max(range(len(matches)), key=lambda i: matches[i][1])
                best_car, best_sim = matches[best_idx]
            else:
                matches = []
                best_car, best_sim = (None, 0.0)
            # Whole-frame interior score for context
            interior_sim = score_interior(img, ref_index, device)
            # Annotated copy
            annotated = draw_annotations(
                img, dets, matches, "lexus_lfa",
                f"{label} · sim={best_sim:.2f}",
            )
            rel = f"annotated/{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{p.stem}.jpg"
            annotated.save(out_dir / rel, quality=85)
            rows.append({
                "label": label,
                "is_positive": is_positive,
                "rel": rel,
                "best_car": best_car or "—",
                "best_sim": float(best_sim),
                "interior_sim": float(interior_sim),
                "n_cars": len(dets),
            })

    # Sort by best_sim descending so the strongest matches surface first
    rows.sort(key=lambda r: r["best_sim"], reverse=True)

    # ----- per-bucket stats -----
    from collections import defaultdict
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["label"]].append(r)

    bucket_rows: list[str] = []
    pos_sims: list[float] = []
    neg_sims: list[float] = []
    for label, _ in [(l, p) for l, p, _ in TEST_BUCKETS]:
        items = grouped.get(label, [])
        sims = [r["best_sim"] for r in items if r["n_cars"] > 0]
        if not sims:
            bucket_rows.append(
                f"<tr><td>{label}</td><td>0</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>"
            )
            continue
        is_pos = next(r["is_positive"] for r in items)
        if is_pos:
            pos_sims.extend(sims)
        else:
            neg_sims.extend(sims)
        n_pass = sum(1 for s in sims if s >= args.target_threshold)
        badge = '<span class="badge pos">LFA</span>' if is_pos else '<span class="badge neg">other</span>'
        bucket_rows.append(
            f"<tr><td>{badge} {label}</td><td>{len(sims)}</td>"
            f"<td>{sum(sims)/len(sims):.3f}</td>"
            f"<td>{max(sims):.3f}</td><td>{min(sims):.3f}</td>"
            f"<td>{n_pass}/{len(sims)}</td></tr>"
        )

    lfa_mean = sum(pos_sims) / max(1, len(pos_sims))
    neg_mean = sum(neg_sims) / max(1, len(neg_sims))
    sep = lfa_mean - neg_mean

    # ----- image cells -----
    cells: list[str] = []
    for r in rows:
        sim = r["best_sim"]
        cls = "high" if sim >= args.target_threshold else "mid" if sim >= 0.40 else "low"
        badge = ('<span class="badge pos">LFA</span>' if r["is_positive"]
                 else '<span class="badge neg">other</span>')
        cells.append(
            f'<div class="cell"><img src="{r["rel"]}">'
            f'<div class="cap">{badge} {r["label"]}<br>'
            f'<span class="score {cls}">sim={sim:.3f}</span>'
            f' · cars={r["n_cars"]} · interior={r["interior_sim"]:.2f}</div></div>'
        )

    html = REPORT_HTML.format(
        lfa_mean=lfa_mean,
        neg_mean=neg_mean,
        sep=sep,
        bucket_rows="\n".join(bucket_rows),
        cells="\n".join(cells),
    )
    (out_dir / "report.html").write_text(html)
    console.print(f"\n[green]wrote[/green] {out_dir / 'report.html'}")
    console.print(f"[bold]LFA mean={lfa_mean:.3f}  non-LFA mean={neg_mean:.3f}  "
                  f"separation={sep:.3f}[/bold]")

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
