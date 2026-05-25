"""Auto-label a YouTube video for a target car.

End-to-end:
  1. Download the video file (mp4) for visual checks AND extract 16kHz wav.
  2. Video-level visual prefilter: sample N frames, run YOLO+DINOv2,
     skip the whole video if the target car appears in too few frames.
  3. Walk overlapping 5s windows. Each window must pass:
       audio gate (silero VAD + CLAP + harmonicity)
       visual gate (sample frames in the window — target visible or empty)
  4. Emit data/regions/<car>/<video_id>.yaml with accepted ranges labeled
     'idle' (placeholder — the state head is unused at training time per
     train.yaml). Rejected stretches become 'skip'.

The output is a drop-in for the existing `carzam window` pipeline.

Notes:
  * We deliberately use a placeholder state label. State classification is
    a separate problem and the user has indicated state weight is 0.0 in
    the current training config.
  * Audio download already exists in download.py. Here we also need the
    video itself for visual frames, so we run yt-dlp once with both
    formats — cheaper than two separate jobs.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from carzam.audio import load_wav, resample_to
from carzam.data.download import target_path_for, video_id_from_url
from carzam.data.regions import Region, regions_path_for, write_regions
from carzam.data.screen import (
    CLAP_SR,
    VAD_SR,
    ScreenThresholds,
    load_clap,
    load_silero_vad,
    screen_window,
)
from carzam.data.visual import (
    ReferenceIndex,
    build_reference_index,
    load_car_detector,
    probe_video_duration,
    screen_video_for_target,
    screen_window_visual,
)

console = Console()


# ---------- download ----------

def _yt_dlp_cmd() -> list[str]:
    """Return the right argv prefix to invoke yt-dlp on this machine.

    Prefers an absolute `yt-dlp` from PATH (covers venv-activated CLI usage),
    falls back to `python -m yt_dlp` so callers from a non-activated venv
    (e.g. a script run via `.venv/bin/python ...`) still work.
    """
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]
    # Try venv-relative
    venv_bin = Path(sys.executable).with_name("yt-dlp")
    if venv_bin.exists():
        return [str(venv_bin)]
    return [sys.executable, "-m", "yt_dlp"]


def _download_video_and_audio(
    url: str, video_out: Path, audio_out: Path,
    audio_only: bool = False,
) -> None:
    """yt-dlp invocation. When audio_only=True, skips the mp4 entirely —
    a 150MB video download per URL is wasteful when the visual gate is off.

    Returns once at least audio_out exists (and video_out too, if not audio_only).
    """
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    ytdlp = _yt_dlp_cmd()

    if not audio_only:
        video_out.parent.mkdir(parents=True, exist_ok=True)
        # 360-480p mp4 is plenty for car detection — keeps download small.
        cmd_video = [
            *ytdlp,
            "-f", "best[height<=480][ext=mp4]/best[ext=mp4]/best",
            "--no-warnings", "--quiet",
            "-o", str(video_out),
            url,
        ]
        subprocess.run(cmd_video, check=True)

    # Audio at 16kHz mono — same defaults as download.py
    tmp_audio = audio_out.with_suffix(".%(ext)s")
    cmd_audio = [
        *ytdlp,
        "-f", "bestaudio",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "--no-warnings", "--quiet",
        "-o", str(tmp_audio),
        url,
    ]
    subprocess.run(cmd_audio, check=True)
    if not audio_out.exists():
        candidates = list(audio_out.parent.glob(f"{audio_out.stem}*.wav"))
        if candidates:
            candidates[0].rename(audio_out)


def _wav_duration(path: Path) -> float:
    """Get duration in seconds from a wav file header (cheap — no full decode)."""
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


# ---------- per-window iteration ----------

@dataclass
class WindowDecision:
    start: float
    end: float
    keep: bool
    audio_clean: bool
    visual_ok: bool
    audio_reasons: list[str]
    visual_reason: str
    clap: dict[str, float]


def _iter_windows(duration: float, window_s: float, hop_s: float):
    """Yield (start, end) covering [0, duration] with `hop_s` stride."""
    t = 0.0
    while t + window_s <= duration:
        yield t, t + window_s
        t += hop_s


# ---------- region collapsing ----------

def _collapse_decisions_to_regions(
    decisions: list[WindowDecision],
    duration: float,
    accept_label: str,
    min_run_seconds: float = 5.0,
) -> list[Region]:
    """Turn per-window keep/skip decisions into contiguous region ranges.

    Adjacent kept windows merge into one 'idle' (or whatever placeholder)
    region; gaps become 'skip'. Drops kept runs shorter than `min_run_seconds`
    so we don't pollute the manifest with one-off windows that might be flukes.
    """
    if not decisions:
        return [Region(0.0, duration, "skip")] if duration > 0 else []

    regions: list[Region] = []
    # Build accept intervals first
    intervals: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float = 0.0
    for d in decisions:
        if d.keep:
            if cur_start is None:
                cur_start = d.start
            cur_end = d.end
        else:
            if cur_start is not None:
                if cur_end - cur_start >= min_run_seconds:
                    intervals.append((cur_start, cur_end))
                cur_start = None
    if cur_start is not None and cur_end - cur_start >= min_run_seconds:
        intervals.append((cur_start, cur_end))

    # Walk the timeline filling skips between intervals
    t = 0.0
    for s, e in intervals:
        if s > t + 0.05:
            regions.append(Region(t, s, "skip"))
        regions.append(Region(s, e, accept_label))
        t = e
    if duration - t > 0.05:
        regions.append(Region(t, duration, "skip"))
    return regions


# ---------- orchestrator ----------

@dataclass
class AutoLabelResult:
    url: str
    car: str
    video_id: str
    duration: float
    video_kept: bool
    video_prefilter: dict | None
    decisions: list[WindowDecision]
    regions: list[Region]
    regions_path: Path | None


def auto_label_video(
    url: str,
    car: str,
    raw_dir: Path,
    regions_dir: Path,
    refs_dir: Path,
    video_cache_dir: Path,
    device: torch.device | None = None,
    window_s: float = 5.0,
    hop_s: float = 2.5,
    n_prefilter_frames: int = 10,
    min_prefilter_target_ratio: float = 0.30,
    n_window_frames: int = 3,
    placeholder_state: str = "idle",
    audio_thresholds: ScreenThresholds = ScreenThresholds(),
    ref_index: ReferenceIndex | None = None,
    keep_video_file: bool = False,
    require_window_visual: bool = True,
) -> AutoLabelResult:
    """Run the full auto-label pipeline on one YouTube URL.

    `ref_index` can be passed in by callers that batch many URLs — avoids
    re-encoding the reference photo library for every video.
    """
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    video_id = video_id_from_url(url)
    audio_path = target_path_for(raw_dir, car, url)
    video_path = video_cache_dir / car / f"{video_id}.mp4"
    regions_path = regions_path_for(regions_dir, car, video_id)

    # Audio-only fast path: when both visual gates are disabled, skip the
    # mp4 download entirely. A 150MB video per URL is wasteful when nothing
    # consumes it.
    audio_only = (
        min_prefilter_target_ratio <= 0.0 and not require_window_visual
    )

    # 1. Models — only load detector when we'll actually use it.
    detector = None
    if not audio_only:
        detector = load_car_detector()
    if ref_index is None and not audio_only:
        console.print(f"[bold]building reference index[/bold] ({refs_dir})")
        ref_index = build_reference_index(refs_dir, device)
        console.print(f"  indexed {len(ref_index.car_per_embedding)} photos "
                      f"across {len(ref_index.cars)} cars")
    # ref_index existence is only enforced when the visual gate runs.
    if ref_index is not None and car not in ref_index.cars and not audio_only:
        raise ValueError(
            f"target car {car!r} has no reference photos under {refs_dir}. "
            f"Add ~15 images to {refs_dir / car}/ first."
        )

    # 2. Download — audio-only when visual is off.
    need_video = (not audio_only) and (not video_path.exists())
    need_audio = not audio_path.exists()
    if need_video or need_audio:
        console.print(f"[bold]downloading[/bold] {'audio only' if audio_only else 'video+audio'} {url}")
        _download_video_and_audio(url, video_path, audio_path, audio_only=audio_only)

    # Duration: from wav header in audio-only, ffprobe-on-mp4 otherwise.
    duration = _wav_duration(audio_path) if audio_only else probe_video_duration(video_path)

    # 3. Video-level visual prefilter — only when we have a video to look at.
    if not audio_only:
        console.print(f"[bold]visual prefilter[/bold] target={car}, "
                      f"{n_prefilter_frames} frames")
        prefilter = screen_video_for_target(
            video_path, car, ref_index, detector, device,
            n_frames=n_prefilter_frames,
            min_target_ratio=min_prefilter_target_ratio,
        )
        console.print(
            f"  target={prefilter['n_target']}/{prefilter['n_frames']}  "
            f"interior={prefilter['n_interior']}  "
            f"other={prefilter['n_other']}  "
            f"unrelated={prefilter['n_empty_unrelated']}  "
            f"majority={prefilter['majority_car']}  "
            f"keep={'yes' if prefilter['keep'] else 'NO'}"
        )
        if not prefilter["keep"]:
            all_skip = [Region(0.0, duration, "skip")] if duration > 0 else []
            write_regions(regions_path, video_id=video_id, duration=duration, regions=all_skip)
            if not keep_video_file and video_path.exists():
                video_path.unlink()
            return AutoLabelResult(
                url=url, car=car, video_id=video_id, duration=duration,
                video_kept=False, video_prefilter=prefilter, decisions=[],
                regions=all_skip, regions_path=regions_path,
            )
    else:
        prefilter = None  # noqa — explicitly skipped in audio-only mode

    # 4. Audio screening models (heavy — loaded only past prefilter)
    console.print("[bold]loading audio models[/bold] (silero VAD + CLAP)")
    vad = load_silero_vad()
    clap_loaded = load_clap(device=device)

    audio, sr = load_wav(audio_path)
    audio_16k = resample_to(audio, src_sr=sr, dst_sr=VAD_SR)
    audio_48k = resample_to(audio, src_sr=sr, dst_sr=CLAP_SR)

    # 5. Per-window walk
    decisions: list[WindowDecision] = []
    n_windows = int(max(0, (duration - window_s) // hop_s + 1))
    console.print(f"[bold]screening {n_windows} windows[/bold]")
    n_kept = 0
    for start, end in _iter_windows(duration, window_s, hop_s):
        i0 = int(start * VAD_SR)
        i1 = int(end * VAD_SR)
        win_16k = audio_16k[i0:i1]
        j0 = int(start * CLAP_SR)
        j1 = int(end * CLAP_SR)
        win_48k = audio_48k[j0:j1]
        if len(win_16k) < int(0.9 * window_s * VAD_SR):
            break

        audio_verdict = screen_window(win_16k, win_48k, vad, clap_loaded, audio_thresholds)
        visual = {"keep": True, "reason": "audio-only" if audio_only else "skipped"}
        audio_ok = audio_verdict.is_clean
        if audio_ok and not audio_only:
            # Only spend ffmpeg/YOLO/DINO cycles on windows that passed audio.
            visual = screen_window_visual(
                video_path, start, window_s, car, ref_index, detector, device,
                n_frames_per_window=n_window_frames,
                require_acceptable=require_window_visual,
            )
        keep = audio_ok and visual["keep"]
        if keep:
            n_kept += 1

        decisions.append(WindowDecision(
            start=start, end=end, keep=keep,
            audio_clean=audio_verdict.is_clean,
            visual_ok=bool(visual["keep"]),
            audio_reasons=audio_verdict.reasons,
            visual_reason=str(visual.get("reason", "")),
            clap=audio_verdict.clap_scores,
        ))

    console.print(f"  kept {n_kept}/{len(decisions)} windows  ({100.0 * n_kept / max(1, len(decisions)):.1f}%)")

    regions = _collapse_decisions_to_regions(decisions, duration, placeholder_state)
    write_regions(regions_path, video_id=video_id, duration=duration, regions=regions)
    accept_secs = sum(r.end - r.start for r in regions if r.label != "skip")
    console.print(
        f"[green]wrote[/green] {regions_path}  "
        f"({accept_secs:.1f}s accepted, {duration - accept_secs:.1f}s skipped)"
    )

    if not keep_video_file and video_path.exists():
        video_path.unlink()

    return AutoLabelResult(
        url=url, car=car, video_id=video_id, duration=duration,
        video_kept=True, video_prefilter=prefilter, decisions=decisions,
        regions=regions, regions_path=regions_path,
    )
