import sys
import termios
import tty
from pathlib import Path

import sounddevice as sd
from rich.console import Console

from carzam.audio import load_wav
from carzam.data.manifest import (
    ManifestRow,
    read_manifest,
    write_manifest,
)

console = Console()

KEY_TO_LABEL = {"i": "idle", "a": "accel", "d": "decel"}


def unlabeled_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    return [r for r in rows if r.label is None]


def _read_key() -> str:
    """Read a single keypress from stdin without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _play(path: Path) -> None:
    audio, sr = load_wav(path)
    sd.play(audio, samplerate=sr)
    sd.wait()


def label_session(manifest_csv: Path) -> None:
    rows = read_manifest(manifest_csv)
    todo = unlabeled_rows(rows)
    console.print(f"[bold]carai labeler[/bold]  {len(todo)} windows pending")
    by_path = {r.path: r for r in rows}

    for i, row in enumerate(todo):
        while True:
            console.print(
                f"\n[{i + 1}/{len(todo)}] [cyan]{row.car}[/cyan] / "
                f"[dim]{row.source_video}[/dim] @ {row.start_time:.1f}s"
            )
            try:
                _play(Path(row.path))
            except Exception as e:
                console.print(f"[red]playback error[/red]: {e} — skipping")
                break
            console.print("  (i)dle  (a)ccel  (d)ecel  (s)kip  (r)eplay  (q)uit")
            key = _read_key().lower()
            if key == "q":
                write_manifest(manifest_csv, rows)
                console.print("[yellow]saved and exiting[/yellow]")
                return
            if key == "r":
                continue
            if key == "s":
                break
            if key in KEY_TO_LABEL:
                by_path[row.path].label = KEY_TO_LABEL[key]
                write_manifest(manifest_csv, rows)
                console.print(f"[green]✓[/green] {KEY_TO_LABEL[key]}")
                break
    write_manifest(manifest_csv, rows)
    console.print("[green]all done[/green]")
