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
def download(sources: Path, out: Path) -> None:
    """Download videos listed in sources.yaml as 16kHz mono WAVs."""
    from carai.data.download import download_all

    download_all(sources, out)


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
def train(config: Path) -> None:
    """Train the model."""
    from carai.train import train as train_run

    train_run(config)


@main.command(name="eval")
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--split", type=click.Choice(["train", "val", "test"]), default="test")
def evaluate_cmd(run_dir: Path, split: str) -> None:
    """Evaluate a trained run on the given split."""
    from carai.eval import evaluate

    evaluate(run_dir, split)


@main.command()
@click.argument("clip")
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True)
@click.option("--json", "as_json", is_flag=True)
def infer(clip: str, checkpoint: Path, as_json: bool) -> None:
    """Predict car + state for a clip (file path or YouTube URL)."""
    import torch

    from carai.infer import predict_clip

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    result = predict_clip(clip, checkpoint, device=device)
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    table = Table(show_header=False)
    table.add_row("clip", str(clip))
    table.add_row("car", f"{result['car']}  ({result['car_confidence']:.2f})")
    for name, p in result["car_top3"]:
        table.add_row("", f"  {name}: {p:.2f}")
    table.add_row("state", f"{result['state']}  ({result['state_confidence']:.2f})")
    console.print(table)
