"""Expand the training dataset with N new supercar/hypercar classes.

For each target car:
  1. Scrape ~15 reference photos from Wikimedia Commons (used to populate
     data/references/<car>/ — the visual gate is currently bypassed, but
     refs are useful for later visual tuning).
  2. ytsearch for ~N candidate YouTube videos (filtered by duration).
  3. Append URLs to config/sources_expanded.yaml.
  4. Run auto_label_video on each — AUDIO-ONLY mode: min_prefilter_target_ratio=0
     and require_window_visual=False. The audio gate (silero VAD + LAION-CLAP
     + harmonicity) does the heavy lifting; the visual side is bypassed
     because our Wikipedia-studio refs don't match YouTube-video distribution.

Idempotent: skips cars whose audio is already downloaded and whose regions
file already exists. Resumable if interrupted.

Output:
  data/references/<car>/*.jpg     # reference photos (for later visual tuning)
  data/raw/<car>/*.wav            # 16kHz mono audio
  data/regions/<car>/*.yaml       # auto-label region ranges
  config/sources_expanded.yaml    # new sources file
  runs/expand_dataset/<ts>/summary.json  # per-car stats

Usage:
  .venv/bin/python scripts/expand_dataset.py --cars pagani_huayra,bugatti_chiron --max-videos 3
  .venv/bin/python scripts/expand_dataset.py --all  # all 50 cars
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich.console import Console

# Reuse the polite Wikimedia scraper from the auto-label demo
sys.path.insert(0, str(Path(__file__).parent))
from demo_auto_label import (  # type: ignore[import-not-found]
    download_images,
    resolve_file_urls,
    search_wikimedia_files,
)

from carzam.data.auto_label import auto_label_video
from carzam.data.download import video_id_from_url
from carzam.data.regions import regions_path_for
from carzam.data.visual import build_reference_index

console = Console()


# car_slug -> (ref_queries, yt_search_phrases)
# ref_queries: search terms for Wikimedia (multiple to maximize photo diversity)
# yt_search_phrases: search terms for ytsearchN (cleaner audio if "pure sound",
# "exhaust", "flyby" are in the phrase)
TARGET_CARS: dict[str, tuple[list[str], list[str]]] = {
    # ----- V12 NA -----
    "pagani_huayra":               (["Pagani Huayra"], ["pagani huayra pure sound", "pagani huayra exhaust"]),
    "pagani_zonda":                (["Pagani Zonda"], ["pagani zonda sound", "pagani zonda exhaust"]),
    "lamborghini_aventador":       (["Lamborghini Aventador"], ["lamborghini aventador sound", "aventador svj exhaust"]),
    "ferrari_enzo":                (["Ferrari Enzo"], ["ferrari enzo sound", "ferrari enzo v12"]),
    "aston_martin_v12_vantage":    (["Aston Martin V12 Vantage"], ["aston martin v12 vantage sound", "v12 vantage exhaust"]),

    # ----- V12 hybrid -----
    "ferrari_laferrari":           (["LaFerrari", "Ferrari LaFerrari"], ["laferrari sound", "laferrari v12"]),
    "aston_martin_valkyrie":       (["Aston Martin Valkyrie"], ["aston martin valkyrie sound", "valkyrie cosworth"]),
    "lamborghini_revuelto":        (["Lamborghini Revuelto"], ["lamborghini revuelto sound", "revuelto v12"]),

    # ----- Classic V8/V10/V12 -----
    "ferrari_f40":                 (["Ferrari F40"], ["ferrari f40 sound", "ferrari f40 exhaust"]),
    "ferrari_f50":                 (["Ferrari F50"], ["ferrari f50 sound", "ferrari f50 v12"]),
    "ferrari_360":                 (["Ferrari 360 Modena", "Ferrari 360"], ["ferrari 360 sound", "ferrari 360 exhaust"]),
    "lamborghini_diablo":          (["Lamborghini Diablo"], ["lamborghini diablo sound", "diablo v12"]),
    "lamborghini_murcielago":      (["Lamborghini Murcielago"], ["murcielago sound", "lp640 sound"]),
    "lamborghini_gallardo":        (["Lamborghini Gallardo"], ["gallardo sound", "gallardo exhaust"]),
    "mclaren_f1":                  (["McLaren F1"], ["mclaren f1 sound", "mclaren f1 v12"]),
    "mclaren_mp4_12c":             (["McLaren MP4-12C", "McLaren 12C"], ["mclaren 12c sound", "mclaren mp4 12c exhaust"]),

    # ----- Modern McLaren -----
    "mclaren_p1":                  (["McLaren P1"], ["mclaren p1 sound", "mclaren p1 hybrid"]),
    "mclaren_speedtail":           (["McLaren Speedtail"], ["mclaren speedtail sound"]),
    "mclaren_artura":              (["McLaren Artura"], ["mclaren artura sound", "artura v6"]),
    "mclaren_750s":                (["McLaren 750S"], ["mclaren 750s sound", "mclaren 750s exhaust"]),

    # ----- Modern Porsche -----
    "porsche_918_spyder":          (["Porsche 918 Spyder"], ["porsche 918 sound", "918 spyder exhaust"]),
    "porsche_carrera_gt":          (["Porsche Carrera GT"], ["carrera gt sound", "carrera gt v10"]),
    "porsche_911_turbo_s":         (["Porsche 911 Turbo S"], ["porsche 911 turbo s sound", "992 turbo s exhaust"]),
    "porsche_992_gt3_rs":          (["Porsche 992 GT3 RS", "Porsche GT3 RS"], ["992 gt3 rs sound", "porsche 992 gt3 rs exhaust"]),
    "porsche_911_gt2_rs":          (["Porsche 911 GT2 RS"], ["911 gt2 rs sound", "porsche gt2 rs exhaust"]),

    # ----- Modern AMG -----
    "mercedes_amg_one":            (["Mercedes-AMG One"], ["amg one sound", "mercedes amg one f1 engine"]),
    "mercedes_amg_gt_black":       (["Mercedes-AMG GT Black Series"], ["amg gt black series sound", "gt black series exhaust"]),
    "mercedes_sls_amg":            (["Mercedes SLS AMG"], ["sls amg sound", "sls amg exhaust"]),
    "mercedes_slr_mclaren":        (["Mercedes-Benz SLR McLaren"], ["slr mclaren sound", "slr mclaren v8"]),

    # ----- British -----
    "lotus_evija":                 (["Lotus Evija"], ["lotus evija sound", "lotus evija electric"]),
    "jaguar_xj220":                (["Jaguar XJ220"], ["jaguar xj220 sound", "xj220 v6"]),
    "aston_dbs_superleggera":      (["Aston Martin DBS Superleggera", "Aston Martin DBS"], ["dbs superleggera sound", "dbs v12 exhaust"]),
    "aston_vanquish":              (["Aston Martin Vanquish"], ["aston vanquish sound", "vanquish v12"]),

    # ----- Japanese -----
    "lexus_lfa":                   (["Lexus LFA"], ["lexus lfa sound", "lexus lfa v10"]),
    "nissan_gtr_r35":              (["Nissan GT-R R35", "Nissan GTR"], ["nissan gtr r35 sound", "r35 gtr exhaust"]),
    "nissan_skyline_r34":          (["Nissan Skyline R34", "Nissan Skyline GT-R R34"], ["r34 gtr sound", "skyline r34 exhaust"]),
    "acura_nsx_nc1":               (["Honda NSX NC1", "Acura NSX"], ["nsx nc1 sound", "acura nsx hybrid sound"]),
    "mazda_rx7_fd":                (["Mazda RX-7 FD"], ["rx7 fd sound", "mazda rx7 rotary exhaust"]),

    # ----- American -----
    "ford_gt_2017":                (["Ford GT 2017"], ["ford gt 2017 sound", "ford gt ecoboost v6"]),
    "ford_gt40":                   (["Ford GT40"], ["ford gt40 sound", "gt40 v8"]),
    "dodge_viper_srt10":           (["Dodge Viper", "Dodge Viper SRT-10"], ["viper srt10 sound", "viper v10 exhaust"]),
    "dodge_demon":                 (["Dodge Challenger SRT Demon"], ["dodge demon sound", "demon supercharged hemi"]),
    "corvette_z06_c8":             (["Chevrolet Corvette Z06 C8", "Corvette C8 Z06"], ["c8 z06 sound", "c8 z06 lt6"]),

    # ----- Hypercars -----
    "bugatti_chiron":              (["Bugatti Chiron"], ["bugatti chiron sound", "chiron w16 exhaust"]),
    "bugatti_veyron":              (["Bugatti Veyron"], ["bugatti veyron sound", "veyron w16"]),
    "koenigsegg_jesko":            (["Koenigsegg Jesko"], ["koenigsegg jesko sound", "jesko exhaust"]),
    "koenigsegg_agera_rs":         (["Koenigsegg Agera RS"], ["agera rs sound", "koenigsegg agera exhaust"]),
    "koenigsegg_regera":           (["Koenigsegg Regera"], ["regera sound", "koenigsegg regera engine"]),
    "rimac_nevera":                (["Rimac Nevera"], ["rimac nevera sound", "rimac nevera electric"]),
    "hennessey_venom_f5":          (["Hennessey Venom F5"], ["hennessey venom f5 sound", "venom f5 v8"]),
}


# ---------- YouTube discovery ----------

def discover_videos(
    queries: list[str], max_videos: int,
    min_duration: float = 30.0, max_duration: float = 900.0,
) -> list[tuple[str, str, float]]:
    """ytsearch each query, return (url, title, duration) tuples, dedup by id."""
    seen_ids: set[str] = set()
    results: list[tuple[str, str, float]] = []
    for q in queries:
        if len(results) >= max_videos:
            break
        cmd = [
            ".venv/bin/yt-dlp", "--dump-json", "--no-warnings", "--skip-download",
            "--match-filter", f"duration > {min_duration} & duration < {max_duration}",
            f"ytsearch{max_videos * 2}:{q}",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            console.print(f"  [yellow]ytsearch timeout[/yellow]: {q}")
            continue
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            vid = d.get("id")
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)
            results.append((
                f"https://youtu.be/{vid}",
                d.get("title", ""),
                float(d.get("duration") or 0),
            ))
            if len(results) >= max_videos:
                break
    return results


# ---------- ensure references ----------

def ensure_car_refs(car_slug: str, ref_queries: list[str], refs_dir: Path,
                    target_count: int = 12) -> int:
    """Idempotent: scrape Wikimedia refs for `car_slug` if not already present.
    Returns total photos in the dir after."""
    car_ref_dir = refs_dir / car_slug
    existing = list(car_ref_dir.glob("*.jpg")) if car_ref_dir.exists() else []
    if len(existing) >= target_count:
        return len(existing)
    car_ref_dir.mkdir(parents=True, exist_ok=True)
    titles: list[str] = []
    for q in ref_queries:
        try:
            titles.extend(search_wikimedia_files(q, limit=20))
        except Exception as e:
            console.print(f"  [yellow]wiki search failed[/yellow] {q!r}: {e}")
    seen: set[str] = set()
    titles = [t for t in titles if not (t in seen or seen.add(t))]
    try:
        urls = resolve_file_urls(titles)
    except Exception as e:
        console.print(f"  [yellow]wiki resolve failed[/yellow]: {e}")
        urls = []
    n = download_images(urls, car_ref_dir, max_n=target_count + 4)
    return n


# ---------- per-car expansion ----------

def expand_one_car(
    car_slug: str,
    ref_queries: list[str],
    yt_queries: list[str],
    refs_dir: Path,
    raw_dir: Path,
    regions_dir: Path,
    video_cache: Path,
    sources_yaml: Path,
    max_videos: int,
    device: torch.device,
    ref_index_holder: list,   # mutable holder so we can rebuild between cars
    scrape_refs: bool = False,
) -> dict:
    """Returns a summary dict for this car."""
    console.print(f"\n[bold cyan]== {car_slug} ==[/bold cyan]")

    # 1. References (skipped in audio-only mode — we don't use them).
    if scrape_refs:
        n_refs = ensure_car_refs(car_slug, ref_queries, refs_dir)
        console.print(f"  refs: {n_refs} photos")
    else:
        n_refs = 0

    # 2. YouTube discovery
    console.print(f"  discovering up to {max_videos} videos...")
    discovered = discover_videos(yt_queries, max_videos)
    console.print(f"  found {len(discovered)} candidates")
    for url, title, dur in discovered:
        console.print(f"    [dim]{int(dur):>4}s[/dim]  {title[:80]}")

    # Persist URLs in sources_expanded.yaml (idempotent)
    sources: dict[str, list[str]] = {}
    if sources_yaml.exists():
        sources = yaml.safe_load(sources_yaml.read_text()) or {}
    existing_urls = set(sources.get(car_slug, []))
    for url, _, _ in discovered:
        existing_urls.add(url)
    sources[car_slug] = sorted(existing_urls)
    sources_yaml.write_text(yaml.safe_dump(sources, sort_keys=True))

    # 3. Reference index — only needed when visual gate runs.
    if scrape_refs:
        try:
            ref_index_holder[0] = build_reference_index(refs_dir, device)
        except Exception as e:
            console.print(f"  [red]ref index rebuild failed[/red]: {e}")
            return {"car": car_slug, "n_refs": n_refs, "n_videos": 0,
                    "n_kept_windows": 0, "n_videos_dropped": 0,
                    "error": str(e)}
    ref_index = ref_index_holder[0]  # may be None in audio-only mode

    # 4. Auto-label each URL — audio-only mode (visual gate effectively off).
    n_kept_total = 0
    n_videos_dropped = 0
    n_videos_processed = 0
    n_video_errors = 0
    for url, title, dur in discovered:
        try:
            vid = video_id_from_url(url)
        except ValueError:
            continue
        rpath = regions_path_for(regions_dir, car_slug, vid)
        if rpath.exists():
            console.print(f"  [dim]skip existing[/dim] {vid}")
            try:
                existing = yaml.safe_load(rpath.read_text())
                kept = sum(
                    r["end"] - r["start"]
                    for r in (existing.get("ranges") or [])
                    if r["label"] != "skip"
                )
                if kept > 0:
                    n_videos_processed += 1
                    n_kept_total += int(kept // 5)
                else:
                    n_videos_dropped += 1
            except Exception:
                pass
            continue
        try:
            result = auto_label_video(
                url=url, car=car_slug,
                raw_dir=raw_dir, regions_dir=regions_dir,
                refs_dir=refs_dir, video_cache_dir=video_cache,
                device=device, ref_index=ref_index,
                # Visual gate disabled — refs don't match video distribution.
                # Audio gate (VAD + CLAP + harmonicity) does the rejection.
                min_prefilter_target_ratio=0.0,
                require_window_visual=False,
            )
            if not result.video_kept:
                n_videos_dropped += 1
            else:
                n_videos_processed += 1
                n_kept_total += sum(1 for d in result.decisions if d.keep)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"  [red]video failed[/red] {url}: {e}")
            console.print(traceback.format_exc(limit=2))
            n_video_errors += 1

    return {
        "car": car_slug,
        "n_refs": n_refs,
        "n_videos_attempted": len(discovered),
        "n_videos_processed": n_videos_processed,
        "n_videos_dropped": n_videos_dropped,
        "n_video_errors": n_video_errors,
        "n_kept_windows": n_kept_total,
    }


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", default=None,
                    help="Comma-separated subset of TARGET_CARS keys "
                         "(default: smoke test of 3 cars).")
    ap.add_argument("--all", action="store_true",
                    help="Run on every car in TARGET_CARS.")
    ap.add_argument("--max-videos", type=int, default=5)
    ap.add_argument("--refs-dir", type=Path, default=Path("data/references"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--regions-dir", type=Path, default=Path("data/regions"))
    ap.add_argument("--video-cache", type=Path, default=Path("data/video_cache"))
    ap.add_argument("--sources-out", type=Path, default=Path("config/sources_expanded.yaml"))
    ap.add_argument("--scrape-refs", action="store_true",
                    help="Also scrape Wikimedia reference photos for each car. "
                         "Not needed in audio-only mode (default off).")
    args = ap.parse_args()

    if args.all:
        cars_to_run = list(TARGET_CARS.keys())
    elif args.cars:
        cars_to_run = [c.strip() for c in args.cars.split(",")]
        for c in cars_to_run:
            if c not in TARGET_CARS:
                console.print(f"[red]unknown car[/red]: {c}")
                console.print(f"  known: {', '.join(TARGET_CARS.keys())}")
                sys.exit(1)
    else:
        # Default smoke test — 3 cars covering different engine families
        cars_to_run = ["pagani_huayra", "aston_martin_v12_vantage", "bugatti_chiron"]

    console.print(f"[bold]expanding[/bold] {len(cars_to_run)} cars: {cars_to_run}")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.print(f"device: {device}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/expand_dataset") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # The auto-label pipeline takes a ref_index. We'll rebuild it after each
    # new car's refs land. Keep it in a mutable holder so expand_one_car
    # can update.
    ref_index_holder: list = [None]

    summary: list[dict] = []
    t_start = time.time()
    for i, car_slug in enumerate(cars_to_run, 1):
        if car_slug not in TARGET_CARS:
            continue
        ref_queries, yt_queries = TARGET_CARS[car_slug]
        try:
            row = expand_one_car(
                car_slug, ref_queries, yt_queries,
                refs_dir=args.refs_dir, raw_dir=args.raw_dir,
                regions_dir=args.regions_dir, video_cache=args.video_cache,
                sources_yaml=args.sources_out,
                max_videos=args.max_videos, device=device,
                ref_index_holder=ref_index_holder,
                scrape_refs=args.scrape_refs,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/yellow]")
            break
        except Exception as e:
            console.print(f"  [red]car failed entirely[/red]: {e}")
            row = {"car": car_slug, "error": str(e),
                   "traceback": traceback.format_exc(limit=3)}
        summary.append(row)
        elapsed = time.time() - t_start
        avg = elapsed / max(1, i)
        remaining = avg * (len(cars_to_run) - i)
        console.print(
            f"[bold green][{i}/{len(cars_to_run)}][/bold green] {car_slug}  "
            f"refs={row.get('n_refs', 0)}  "
            f"kept_windows={row.get('n_kept_windows', 0)}  "
            f"({elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m left)"
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Final summary
    total_kept = sum(r.get("n_kept_windows", 0) for r in summary)
    total_videos = sum(r.get("n_videos_processed", 0) for r in summary)
    total_dropped = sum(r.get("n_videos_dropped", 0) for r in summary)
    console.print(f"\n[bold green]done[/bold green] ({(time.time()-t_start)/60:.1f}m)")
    console.print(f"  cars processed: {len(summary)}")
    console.print(f"  videos kept: {total_videos}, dropped: {total_dropped}")
    console.print(f"  total auto-labeled windows: {total_kept}")
    console.print(f"  sources written to: {args.sources_out}")
    console.print(f"  summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
