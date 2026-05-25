"""Window all auto-labeled regions and retrain the contrastive model.

After expand_dataset.py finishes, run this to:
  1. carzam window — slice all regions into 5s training windows + rebuild manifest
  2. Train each variant config in series (v2, v3, v4)
  3. Run compare_runs.py to evaluate them side by side

Each training step writes to runs/<ts>/. The compare step picks them all up
and renders a single HTML report.

Usage:
  .venv/bin/python scripts/window_and_train.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_CONFIGS = [
    "config/train_contrastive_v2_warm.yaml",
    "config/train_contrastive_v3_lowlr.yaml",
    "config/train_contrastive_v4_widebatch.yaml",
]


def run_step(cmd: list[str], description: str) -> bool:
    console.print(f"\n[bold cyan]>>> {description}[/bold cyan]")
    console.print(f"    {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        console.print(f"[red]step failed[/red] (rc={rc})")
        return False
    return True


def latest_run_dir(prev_count: int) -> Path | None:
    """Return the newest dir in runs/ if it's newer than prev_count snapshot."""
    runs = sorted(Path("runs").glob("20*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--skip-window", action="store_true",
                    help="Skip the window step — assume manifest is already current.")
    args = ap.parse_args()

    if not args.skip_window:
        ok = run_step(
            [".venv/bin/carzam", "window"],
            "1. window: re-slice all regions into 5s clips and rebuild manifest",
        )
        if not ok:
            sys.exit(1)

    started_at = datetime.now()
    run_dirs: list[Path] = []
    for cfg in args.configs:
        before = len(list(Path("runs").glob("20*")))
        ok = run_step(
            [".venv/bin/carzam", "train", "--config", cfg],
            f"train: {Path(cfg).name}",
        )
        if not ok:
            console.print(f"[yellow]continuing despite failure on {cfg}[/yellow]")
            continue
        new_run = latest_run_dir(before)
        if new_run:
            run_dirs.append(new_run)
            console.print(f"  → {new_run}")

    if run_dirs:
        runs_arg = ",".join(str(p) for p in run_dirs)
        run_step(
            [".venv/bin/python", "scripts/compare_runs.py",
             "--runs", runs_arg, "--no-open"],
            "compare: render HTML report",
        )
    console.print(f"\n[bold green]done.[/bold green] total time: "
                  f"{(datetime.now() - started_at).total_seconds() / 60:.1f} min")


if __name__ == "__main__":
    main()
