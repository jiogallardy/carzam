import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def main() -> None:
    """carai — car engine sound classifier"""


@main.command()
@click.option("--sources", type=click.Path(path_type=Path), default=Path("config/sources.yaml"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--regions", type=click.Path(path_type=Path), default=Path("data/regions"),
              help="Only download URLs that have a regions YAML (= approved in review).")
@click.option("--all-sources", is_flag=True,
              help="Download every URL in sources.yaml regardless of approval state.")
def download(sources: Path, out: Path, regions: Path, all_sources: bool) -> None:
    """Download videos listed in sources.yaml as 16kHz mono WAVs.

    By default, only approved videos (those with a regions YAML) are pulled.
    Pass --all-sources to grab everything in sources.yaml.
    """
    from carai.data.download import download_all
    from carai.data.regions import regions_path_for

    filter_fn = None
    if not all_sources:
        def is_approved(car: str, url: str) -> bool:
            from carai.data.download import video_id_from_url
            try:
                vid = video_id_from_url(url)
            except ValueError:
                return False
            return regions_path_for(regions, car, vid).exists()
        filter_fn = is_approved

    download_all(sources, out, filter_fn=filter_fn)


@main.command()
@click.option("--raw", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/windows"))
@click.option("--manifest", type=click.Path(path_type=Path), default=Path("data/manifest.csv"))
@click.option("--regions", type=click.Path(path_type=Path), default=Path("data/regions"))
@click.option("--rms-threshold", type=float, default=0.005)
def window(raw: Path, out: Path, manifest: Path, regions: Path, rms_threshold: float) -> None:
    """Slice raw WAVs into 5-second windows. Uses regions when present, RMS gate otherwise."""
    from carai.data.manifest import ManifestRow, write_manifest
    from carai.data.regions import read_regions, regions_path_for
    from carai.data.window import slice_into_windows, slice_with_regions

    rows: list[ManifestRow] = []
    used_regions = used_rms = 0
    for car_dir in sorted(raw.iterdir()):
        if not car_dir.is_dir():
            continue
        car = car_dir.name
        car_out = out / car
        for src in sorted(car_dir.glob("*.wav")):
            rpath = regions_path_for(regions, car, src.stem)
            if rpath.exists():
                try:
                    _, _, regs = read_regions(rpath)
                except Exception as e:
                    console.print(f"[red]bad regions file {rpath}: {e}[/red]")
                    continue
                if not regs:
                    console.print(f"[dim]skip[/dim] {src.name} (empty regions)")
                    continue
                console.print(f"[bold green]regions[/bold green] {src.name}")
                wins = slice_with_regions(src, car_out, regs, 5.0, 2.5)
                used_regions += 1
                for w in wins:
                    rows.append(
                        ManifestRow(
                            path=str(w.path),
                            car=car,
                            source_video=w.source_video,
                            start_time=w.start_time,
                            label=w.label,
                        )
                    )
            else:
                console.print(f"[dim]rms[/dim] {src.name}")
                wins = slice_into_windows(src, car_out, 5.0, 2.5, rms_threshold)
                used_rms += 1
                for w in wins:
                    rows.append(
                        ManifestRow(
                            path=str(w.path),
                            car=car,
                            source_video=w.source_video,
                            start_time=w.start_time,
                            label=None,
                        )
                    )
    write_manifest(manifest, rows)
    labeled = sum(1 for r in rows if r.label is not None)
    console.print(
        f"[green]wrote {len(rows)} windows[/green] "
        f"({labeled} labeled from regions, {len(rows) - labeled} unlabeled)  "
        f"-> {manifest}  (used regions on {used_regions} videos, RMS on {used_rms})"
    )


@main.command()
@click.option("--raw", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--regions", type=click.Path(path_type=Path), default=Path("data/regions"))
@click.option("--speed", type=float, default=1.0, help="Playback speed (1.0–3.0).")
@click.option("--video", type=click.Path(path_type=Path), default=None,
              help="Review only this single WAV instead of all undone videos.")
def review(raw: Path, regions: Path, speed: float, video: Path | None) -> None:
    """Terminal-based reviewer (audio-only). For most cases use 'review-web' instead."""
    from carai.data.regions import regions_path_for
    from carai.data.reviewer import review_session, review_video

    if video is not None:
        car = video.parent.name
        rpath = regions_path_for(regions, car, video.stem)
        review_video(video, rpath, speed=speed)
    else:
        review_session(raw, regions, speed=speed)


@main.command()
@click.argument("audio_path", type=click.Path(path_type=Path, exists=True))
@click.option("--start", type=float, default=0.0,
              help="Start time in seconds (anything before is marked as skip).")
@click.option("--end", type=float, default=None,
              help="End time in seconds (default: full duration).")
@click.option("--label", type=click.Choice(["idle", "accel", "decel"]), default="idle",
              help="Placeholder state label. State head is unused at training time, "
                   "so this rarely matters — defaults to 'idle'.")
@click.option("--regions", type=click.Path(path_type=Path), default=Path("data/regions"))
def mark(audio_path: Path, start: float, end: float | None, label: str, regions: Path) -> None:
    """Bulk-mark a downloaded audio file with a single labeled range.

    Use when you've already identified the audio as a particular car and don't
    want to interactively label idle/accel/decel sections. Creates a regions
    YAML covering [start, end] as the given placeholder state. Anything outside
    that range becomes skip. The class comes from the parent directory name.

    Examples:
      # whole file as one labeled chunk
      carai mark data/raw/porsche_gt4/QIIO2SOSQMw.wav

      # only seconds 28..143 are usable, skip the rest
      carai mark data/raw/porsche_gt3/IUN-bVQ-KXI.wav --start 28 --end 143
    """
    from carai.audio import load_wav
    from carai.data.regions import Region, regions_path_for, write_regions

    car = audio_path.parent.name
    video_id = audio_path.stem
    audio, sr = load_wav(audio_path)
    duration = len(audio) / sr

    if end is None or end > duration:
        end = duration
    if start < 0:
        start = 0.0
    if end <= start:
        raise click.UsageError(f"end ({end}) must be > start ({start})")

    ranges: list[Region] = []
    if start > 0.05:
        ranges.append(Region(start=0.0, end=start, label="skip"))
    ranges.append(Region(start=start, end=end, label=label))
    if duration - end > 0.05:
        ranges.append(Region(start=end, end=duration, label="skip"))

    rpath = regions_path_for(regions, car, video_id)
    write_regions(rpath, video_id=video_id, duration=duration, regions=ranges)
    console.print(
        f"[green]marked[/green] {car}/{video_id}: "
        f"{start:.1f}s–{end:.1f}s as [bold]{label}[/bold]  ({duration:.1f}s total)  -> {rpath}"
    )


@main.command(name="auto-label")
@click.argument("url")
@click.option("--car", required=True, help="Target car class — must have reference photos under --refs/<car>/")
@click.option("--raw", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--regions", type=click.Path(path_type=Path), default=Path("data/regions"))
@click.option("--refs", type=click.Path(path_type=Path), default=Path("data/references"))
@click.option("--video-cache", type=click.Path(path_type=Path), default=Path("data/video_cache"))
@click.option("--keep-video", is_flag=True, help="Don't delete the mp4 after labeling (useful for debugging).")
@click.option("--min-target-ratio", type=float, default=0.30,
              help="Fraction of non-empty prefilter frames that must match target car. Below this, the whole video is dropped.")
def auto_label_cmd(
    url: str, car: str, raw: Path, regions: Path, refs: Path,
    video_cache: Path, keep_video: bool, min_target_ratio: float,
) -> None:
    """Auto-label one YouTube video for the given car.

    Downloads the video + audio, runs a video-level visual prefilter using
    YOLOv8 + DINOv2 against reference photos in --refs/<car>/, then walks
    5s windows applying VAD + CLAP audio gates and per-window visual checks.
    Emits a regions.yaml drop-in for `carai window`.
    """
    import torch

    from carai.data.auto_label import auto_label_video

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    auto_label_video(
        url=url, car=car,
        raw_dir=raw, regions_dir=regions, refs_dir=refs,
        video_cache_dir=video_cache,
        device=device,
        min_prefilter_target_ratio=min_target_ratio,
        keep_video_file=keep_video,
    )


@main.command(name="auto-label-batch")
@click.option("--sources", type=click.Path(path_type=Path), default=Path("config/sources.yaml"),
              help="Same format as the manual pipeline — car name → list of YouTube URLs.")
@click.option("--cars", default=None,
              help="Comma-separated subset of cars to process (default: all cars in sources).")
@click.option("--raw", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--regions", type=click.Path(path_type=Path), default=Path("data/regions"))
@click.option("--refs", type=click.Path(path_type=Path), default=Path("data/references"))
@click.option("--video-cache", type=click.Path(path_type=Path), default=Path("data/video_cache"))
@click.option("--keep-video", is_flag=True)
@click.option("--skip-existing", is_flag=True,
              help="Skip URLs that already have a regions.yaml.")
def auto_label_batch_cmd(
    sources: Path, cars: str | None, raw: Path, regions: Path, refs: Path,
    video_cache: Path, keep_video: bool, skip_existing: bool,
) -> None:
    """Run auto-label across every URL in sources.yaml. Builds the reference
    index once and reuses it for all videos."""
    import torch
    import yaml

    from carai.data.auto_label import auto_label_video
    from carai.data.download import video_id_from_url
    from carai.data.regions import regions_path_for
    from carai.data.visual import build_reference_index

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    src = yaml.safe_load(Path(sources).read_text())

    if cars:
        wanted = {c.strip() for c in cars.split(",")}
        src = {k: v for k, v in src.items() if k in wanted}

    console.print("[bold]building reference index[/bold]")
    ref_index = build_reference_index(refs, device)
    console.print(f"  indexed {len(ref_index.car_per_embedding)} photos "
                  f"across {len(ref_index.cars)} cars")

    n_total = sum(len(v) for v in src.values())
    n_done = 0
    n_kept = 0
    for car, urls in src.items():
        if car not in ref_index.cars:
            console.print(f"[yellow]no reference photos for {car} — skipping its videos[/yellow]")
            n_done += len(urls)
            continue
        for url in urls:
            n_done += 1
            try:
                vid = video_id_from_url(url)
            except ValueError:
                console.print(f"[red]bad url[/red] {url}")
                continue
            if skip_existing and regions_path_for(regions, car, vid).exists():
                console.print(f"[dim]skip existing[/dim] {car}/{vid}")
                continue
            console.print(f"\n[bold cyan]\\[{n_done}/{n_total}] {car}[/bold cyan] {url}")
            try:
                result = auto_label_video(
                    url=url, car=car,
                    raw_dir=raw, regions_dir=regions, refs_dir=refs,
                    video_cache_dir=video_cache, device=device,
                    ref_index=ref_index, keep_video_file=keep_video,
                )
                if result.video_kept:
                    n_kept += 1
            except Exception as e:
                console.print(f"[red]failed[/red] {url}: {e}")
    console.print(f"\n[green]done[/green] — kept {n_kept}/{n_total} videos")


@main.command(name="visual-check")
@click.argument("video", type=click.Path(path_type=Path, exists=True))
@click.option("--car", required=True, help="Target car to check against.")
@click.option("--refs", type=click.Path(path_type=Path), default=Path("data/references"))
@click.option("--n-frames", type=int, default=10)
def visual_check_cmd(video: Path, car: str, refs: Path, n_frames: int) -> None:
    """Stand-alone visual prefilter on an existing video file. Prints what
    the YOLO+DINOv2 stack thinks of each sampled frame — useful for
    sanity-checking the reference photo library before running auto-label."""
    import torch

    from carai.data.visual import build_reference_index, load_car_detector, screen_video_for_target

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ref_index = build_reference_index(refs, device)
    detector = load_car_detector()
    if car not in ref_index.cars:
        raise click.UsageError(f"no reference photos for {car!r} under {refs}/{car}/")
    result = screen_video_for_target(video, car, ref_index, detector, device, n_frames=n_frames)

    table = Table(title=f"visual check: {video.name} (target={car})")
    table.add_column("ts", justify="right")
    table.add_column("cars")
    table.add_column("best match")
    table.add_column("score", justify="right")
    table.add_column("interior", justify="right")
    table.add_column("verdict")
    for ts, v in zip(result["timestamps"], result["verdicts"]):
        if v.is_target:
            verdict = "[green]target[/green]"
        elif v.is_empty and v.is_interior:
            verdict = "[cyan]interior[/cyan]"
        elif v.is_empty:
            verdict = "[dim]unrelated[/dim]"
        elif v.other_car_dominant:
            verdict = "[red]other[/red]"
        else:
            verdict = "[yellow]weak[/yellow]"
        table.add_row(
            f"{ts:.1f}s",
            str(v.n_cars_detected),
            v.best_match_car or "-",
            f"{v.best_match_score:.2f}",
            f"{v.interior_score:.2f}" if v.is_empty else "-",
            verdict,
        )
    console.print(table)
    console.print(
        f"target={result['target_ratio']:.2f}  "
        f"interior={result['interior_ratio']:.2f}  "
        f"acceptable={result['acceptable_ratio']:.2f}  "
        f"other={result['other_ratio']:.2f}  "
        f"majority={result['majority_car']}  "
        f"=> [{'green' if result['keep'] else 'red'}]"
        f"{'KEEP' if result['keep'] else 'DROP'}[/]"
    )


@main.command(name="review-web")
@click.option("--sources", type=click.Path(path_type=Path), default=Path("config/sources.yaml"))
@click.option("--regions", type=click.Path(path_type=Path), default=Path("data/regions"))
@click.option("--port", type=int, default=7777)
@click.option("--no-open", is_flag=True, help="Don't auto-open the browser.")
def review_web(sources: Path, regions: Path, port: int, no_open: bool) -> None:
    """Browser-based reviewer: streams YouTube videos, captures region marks."""
    from carai.data.web_reviewer import serve

    serve(sources, regions, port=port, open_browser=not no_open)


@main.command()
@click.option("--manifest", type=click.Path(path_type=Path), default=Path("data/manifest.csv"))
def label(manifest: Path) -> None:
    """Interactive labeler. Resumes where you left off."""
    from carai.data.labeler import label_session

    label_session(manifest)


@main.command()
@click.option("--config", type=click.Path(path_type=Path), default=Path("config/train.yaml"))
@click.option("--family", default=None,
              help="Specialist mode: train only on cars in this engine family "
                   "(e.g. na_flat6, na_v12). Overrides the config's specialist_family.")
def train(config: Path, family: str | None) -> None:
    """Train the model. Use --family to train a v8 cascade specialist."""
    from carai.train import train as train_run

    train_run(config, specialist_family_override=family)


@main.command(name="eval")
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--split", type=click.Choice(["train", "val", "test"]), default="test")
def evaluate_cmd(run_dir: Path, split: str) -> None:
    """Evaluate a trained run on the given split."""
    from carai.eval import evaluate

    evaluate(run_dir, split)


@main.command()
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True)
@click.option("--duration", type=float, default=5.0, help="Seconds to record (default 5).")
@click.option("--json", "as_json", is_flag=True)
def listen(checkpoint: Path, duration: float, as_json: bool) -> None:
    """Record from default mic for N seconds, then predict the car. Shazam-style."""
    import torch

    from carai.infer import listen_and_predict

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.print(f"[bold cyan]listening for {duration:.1f}s...[/bold cyan]")
    result = listen_and_predict(checkpoint, duration=duration, device=device)
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    table = Table(show_header=False)
    table.add_row("source", "microphone")
    table.add_row("duration", f"{duration:.1f}s")
    table.add_row("car", f"{result['car']}  ({result['car_confidence']:.2f})")
    for name, p in result["car_top3"]:
        table.add_row("", f"  {name}: {p:.2f}")
    table.add_row("state", f"{result['state']}  ({result['state_confidence']:.2f})")
    console.print(table)


@main.command()
@click.argument("clip")
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True)
@click.option("--start", type=float, default=None,
              help="Start time in seconds (clip the input before predicting).")
@click.option("--duration", type=float, default=None,
              help="Duration in seconds (default: full clip). Use 5 for Shazam-style.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--timeline", is_flag=True, help="Print per-window predictions for long clips.")
def infer(
    clip: str, checkpoint: Path, start: float | None, duration: float | None,
    as_json: bool, timeline: bool,
) -> None:
    """Predict car + state for a clip (file path or YouTube URL).

    Long clips are auto-sliced into overlapping 5s windows; predictions are
    averaged for the verdict. Use --timeline to also see per-window results.
    Use --start S --duration 5 to grab a single 5s slice (Shazam-style).
    """
    import torch

    from carai.infer import predict_clip

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    result = predict_clip(clip, checkpoint, device=device, start=start, duration=duration)
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    n_windows = result.get("n_windows", 1)
    table = Table(show_header=False)
    table.add_row("clip", str(clip))
    table.add_row("windows", f"{n_windows}")
    table.add_row("car", f"{result['car']}  ({result['car_confidence']:.2f})")
    for name, p in result["car_top3"]:
        table.add_row("", f"  {name}: {p:.2f}")
    table.add_row("state", f"{result['state']}  ({result['state_confidence']:.2f})")
    console.print(table)

    if timeline and n_windows > 1:
        tl = Table(title="per-window timeline")
        tl.add_column("start", justify="right")
        tl.add_column("car")
        tl.add_column("conf", justify="right")
        tl.add_column("state")
        for w in result["windows"]:
            tl.add_row(
                f"{w['start']:.1f}s",
                w["car"],
                f"{w['car_confidence']:.2f}",
                w["state"],
            )
        console.print(tl)
