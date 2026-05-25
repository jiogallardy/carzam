"""Interactive region reviewer.

Plays the audio of a source video, captures keystrokes that toggle the current
label mode (idle/accel/decel/skip). Saves a regions YAML when done.
"""
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console

from carzam.audio import load_wav
from carzam.data.regions import (
    REGION_LABELS,
    Region,
    read_regions,
    regions_path_for,
    write_regions,
)

console = Console()

KEY_TO_MODE = {"i": "idle", "a": "accel", "d": "decel", "s": "skip"}


def _setup_raw_stdin() -> tuple[int, list]:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return fd, old


def _restore_stdin(fd: int, old: list) -> None:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_nonblocking(timeout: float = 0.05) -> str | None:
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def _format_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:d}:{s:02d}"


def review_video(
    wav_path: Path | str,
    regions_path: Path | str,
    speed: float = 1.0,
) -> bool:
    """Open `wav_path`, capture region marks, save to `regions_path`.

    Returns True if regions were saved, False if user pressed `n` (next/skip
    saving) or quit before any input.
    """
    wav_path = Path(wav_path)
    regions_path = Path(regions_path)

    audio, sr = load_wav(wav_path)
    duration_orig = len(audio) / sr

    # Apply speed via simple resample of the playback buffer (changes pitch slightly,
    # but for review purposes the discriminating cues survive). No external dep.
    if speed != 1.0:
        new_len = int(len(audio) / speed)
        idx = np.linspace(0, len(audio) - 1, new_len).astype(np.int64)
        audio_play = audio[idx]
    else:
        audio_play = audio

    # Mutable state shared between callback + main loop
    pos_samples = [0]
    is_playing = [True]
    current_mode: list[str | None] = [None]
    mode_start_orig: list[float] = [0.0]
    finished_ranges: list[Region] = []

    def callback(outdata, frames, time_info, status):
        if not is_playing[0]:
            outdata[:] = 0
            return
        s = pos_samples[0]
        e = s + frames
        if e > len(audio_play):
            chunk = audio_play[s : len(audio_play)]
            outdata[: len(chunk), 0] = chunk
            outdata[len(chunk) :, 0] = 0
            pos_samples[0] = len(audio_play)
            raise sd.CallbackStop
        outdata[:, 0] = audio_play[s:e]
        pos_samples[0] = e

    def current_orig_time() -> float:
        return min(duration_orig, (pos_samples[0] / sr) * speed)

    def switch_mode(new_mode: str | None) -> None:
        now = current_orig_time()
        if current_mode[0] is not None:
            if now > mode_start_orig[0] + 0.01:  # ignore zero-length toggles
                finished_ranges.append(Region(mode_start_orig[0], now, current_mode[0]))
        current_mode[0] = new_mode
        mode_start_orig[0] = now

    stream = sd.OutputStream(samplerate=sr, channels=1, callback=callback)
    fd, old = _setup_raw_stdin()
    aborted = False
    save = True

    try:
        # Header (printed before raw mode swallows formatting)
        sys.stdout.write(
            f"\r\033[K[reviewing] {wav_path.name}  duration {_format_time(duration_orig)}  speed {speed}x\r\n"
        )
        sys.stdout.write("  i=idle  a=accel  d=decel  s=skip  x=close  space=pause  z/c=±5s  u=undo  q=save&quit  n=next-no-save\r\n")
        sys.stdout.flush()

        stream.start()
        last_render = 0.0
        while True:
            now = current_orig_time()
            wall_now = time.monotonic()
            if wall_now - last_render > 0.1:
                last_render = wall_now
                bar_n = 40
                progress = int((now / duration_orig) * bar_n) if duration_orig > 0 else 0
                bar = "█" * progress + "·" * (bar_n - progress)
                mode_str = current_mode[0] if current_mode[0] else "—"
                sys.stdout.write(
                    f"\r\033[K  {_format_time(now)}/{_format_time(duration_orig)}  "
                    f"[{bar}]  mode={mode_str:<6}  ranges={len(finished_ranges)}"
                )
                sys.stdout.flush()

            key = _read_key_nonblocking(0.05)
            if key is None:
                if pos_samples[0] >= len(audio_play):
                    break
                continue
            k = key.lower()
            if k in KEY_TO_MODE:
                switch_mode(KEY_TO_MODE[k])
            elif k == "x":
                switch_mode(None)
            elif k == " ":
                is_playing[0] = not is_playing[0]
            elif k == "z":
                # rewind 5s in original time -> stretched samples
                back = int((5.0 / speed) * sr)
                pos_samples[0] = max(0, pos_samples[0] - back)
            elif k == "c":
                fwd = int((5.0 / speed) * sr)
                pos_samples[0] = min(len(audio_play), pos_samples[0] + fwd)
            elif k == "u" and finished_ranges:
                last = finished_ranges.pop()
                # reopen its mode from where we are, anchored at last.start
                current_mode[0] = last.label
                mode_start_orig[0] = last.start
            elif k == "q":
                if current_mode[0] is not None:
                    switch_mode(None)
                break
            elif k == "n":
                save = False
                break
            elif k == "\x03":  # Ctrl-C
                aborted = True
                save = False
                break
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        _restore_stdin(fd, old)
        sys.stdout.write("\r\033[K\n")
        sys.stdout.flush()

    if aborted:
        console.print("[red]aborted[/red]")
        raise KeyboardInterrupt

    if not save:
        console.print("[yellow]skipped (no save)[/yellow]")
        return False

    # close any still-open mode
    if current_mode[0] is not None:
        finished_ranges.append(
            Region(mode_start_orig[0], current_orig_time(), current_mode[0])
        )

    # de-duplicate adjacent same-label ranges
    merged: list[Region] = []
    for r in finished_ranges:
        if merged and merged[-1].label == r.label and abs(merged[-1].end - r.start) < 0.05:
            merged[-1] = Region(merged[-1].start, r.end, r.label)
        else:
            merged.append(r)

    write_regions(regions_path, video_id=wav_path.stem, duration=duration_orig, regions=merged)
    by_label = {lbl: 0.0 for lbl in REGION_LABELS}
    for r in merged:
        by_label[r.label] += r.end - r.start
    summary = "  ".join(f"{lbl}:{by_label[lbl]:.0f}s" for lbl in REGION_LABELS)
    console.print(f"[green]saved {len(merged)} regions[/green]  {summary}  -> {regions_path}")
    return True


def review_session(
    raw_dir: Path | str,
    regions_dir: Path | str,
    speed: float = 1.0,
) -> None:
    """Iterate over all undone (car, video) pairs, prompt review for each."""
    raw_dir = Path(raw_dir)
    regions_dir = Path(regions_dir)

    todo: list[tuple[str, Path, Path]] = []
    for car_dir in sorted(raw_dir.iterdir()):
        if not car_dir.is_dir():
            continue
        car = car_dir.name
        for src in sorted(car_dir.glob("*.wav")):
            rpath = regions_path_for(regions_dir, car, src.stem)
            if rpath.exists():
                # Skip if already saved with non-empty ranges
                try:
                    _, _, regs = read_regions(rpath)
                    if regs:
                        continue
                except Exception:
                    pass
            todo.append((car, src, rpath))

    if not todo:
        console.print("[green]nothing to review — all videos already have regions[/green]")
        return

    console.print(f"[bold]{len(todo)}[/bold] videos pending review")
    for i, (car, src, rpath) in enumerate(todo, 1):
        console.print(f"\n[bold cyan]({i}/{len(todo)}) {car}[/bold cyan]  {src.name}")
        try:
            review_video(src, rpath, speed=speed)
        except KeyboardInterrupt:
            console.print("[red]interrupted, exiting[/red]")
            return
