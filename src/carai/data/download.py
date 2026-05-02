import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
from rich.console import Console

console = Console()


def video_id_from_url(url: str) -> str:
    """Extract YouTube video ID from a watch?v= or youtu.be/ URL."""
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.lstrip("/")
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    raise ValueError(f"Cannot extract video id from {url!r}")


def target_path_for(raw_dir: Path | str, car: str, url: str) -> Path:
    return Path(raw_dir) / car / f"{video_id_from_url(url)}.wav"


def is_already_downloaded(raw_dir: Path | str, car: str, url: str) -> bool:
    return target_path_for(raw_dir, car, url).exists()


def download_one(url: str, out_path: Path) -> None:
    """Download audio-only as 16kHz mono WAV via yt-dlp."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_template = str(out_path.with_suffix(".%(ext)s"))
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "-o", tmp_template,
        url,
    ]
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        candidates = list(out_path.parent.glob(f"{out_path.stem}*.wav"))
        if candidates:
            candidates[0].rename(out_path)


def download_all(sources_yaml: Path, raw_dir: Path) -> None:
    sources = yaml.safe_load(Path(sources_yaml).read_text())
    for car, urls in sources.items():
        for url in urls:
            if is_already_downloaded(raw_dir, car, url):
                console.print(f"[dim]skip[/dim] {car} {url}")
                continue
            out_path = target_path_for(raw_dir, car, url)
            console.print(f"[bold]dl[/bold] {car} {url} -> {out_path}")
            try:
                download_one(url, out_path)
            except subprocess.CalledProcessError as e:
                console.print(f"[red]fail[/red] {url}: {e}")
