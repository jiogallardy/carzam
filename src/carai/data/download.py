import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
from rich.console import Console

console = Console()

# yt-dlp throttling — keeps YouTube from issuing 429s on bulk runs
SLEEP_REQUESTS = "1.5"     # seconds between metadata requests
SLEEP_INTERVAL_MIN = "3"   # min seconds between downloads
SLEEP_INTERVAL_MAX = "8"   # max seconds (yt-dlp picks uniformly)
RETRY_BACKOFF_SECONDS = (15, 60, 180)  # waits between retries on failure


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
    """Download audio-only as 16kHz mono WAV via yt-dlp, with built-in throttling."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_template = str(out_path.with_suffix(".%(ext)s"))
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "--sleep-requests", SLEEP_REQUESTS,
        "--sleep-interval", SLEEP_INTERVAL_MIN,
        "--max-sleep-interval", SLEEP_INTERVAL_MAX,
        "--retries", "5",
        "--retry-sleep", "linear=10:60",
        "-o", tmp_template,
        url,
    ]
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        candidates = list(out_path.parent.glob(f"{out_path.stem}*.wav"))
        if candidates:
            candidates[0].rename(out_path)


def download_all(
    sources_yaml: Path,
    raw_dir: Path,
    filter_fn: "callable | None" = None,
) -> None:
    sources = yaml.safe_load(Path(sources_yaml).read_text())
    failed: list[tuple[str, str]] = []
    for car, urls in sources.items():
        for url in urls:
            if filter_fn is not None and not filter_fn(car, url):
                console.print(f"[dim]not approved[/dim] {car} {url}")
                continue
            if is_already_downloaded(raw_dir, car, url):
                console.print(f"[dim]skip[/dim] {car} {url}")
                continue
            out_path = target_path_for(raw_dir, car, url)
            console.print(f"[bold]dl[/bold] {car} {url} -> {out_path}")

            success = False
            for attempt, wait in enumerate([0, *RETRY_BACKOFF_SECONDS], start=1):
                if wait:
                    console.print(f"  [yellow]retry {attempt} after {wait}s[/yellow]")
                    time.sleep(wait)
                try:
                    download_one(url, out_path)
                    success = True
                    break
                except subprocess.CalledProcessError as e:
                    console.print(f"  [red]attempt {attempt} failed[/red]: {e}")

            if not success:
                console.print(f"[red]GAVE UP[/red] {url}")
                failed.append((car, url))

    if failed:
        console.print(f"\n[red]{len(failed)} videos failed after retries:[/red]")
        for car, url in failed:
            console.print(f"  {car}: {url}")
    else:
        console.print("\n[green]all videos downloaded[/green]")
