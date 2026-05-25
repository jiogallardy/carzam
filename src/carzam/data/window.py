from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from carzam.audio import load_wav
from carzam.data.regions import Region, label_for_time


@dataclass
class Window:
    path: Path
    source_video: str
    start_time: float
    label: str | None = None  # set when sliced with regions


def slice_into_windows(
    src_wav: Path | str,
    out_dir: Path | str,
    window_seconds: float = 5.0,
    hop_seconds: float = 2.5,
    rms_threshold: float = 0.005,
) -> list[Window]:
    """Original silence-gated slicer. Used as fallback when no regions exist."""
    src_wav = Path(src_wav)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = load_wav(src_wav)
    win_len = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    source_video = src_wav.stem

    out: list[Window] = []
    start = 0
    while start + win_len <= len(audio):
        chunk = audio[start : start + win_len]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms >= rms_threshold:
            start_time = start / sr
            out_path = out_dir / f"{source_video}_{start_time:07.2f}.wav"
            sf.write(out_path, chunk, sr)
            out.append(Window(out_path, source_video, start_time))
        start += hop
    return out


def slice_with_regions(
    src_wav: Path | str,
    out_dir: Path | str,
    regions: list[Region],
    window_seconds: float = 5.0,
    hop_seconds: float = 2.5,
) -> list[Window]:
    """Region-driven slicer. Emits windows that fit entirely inside a non-skip
    region; each window inherits that region's label.
    """
    src_wav = Path(src_wav)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = load_wav(src_wav)
    win_len = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    source_video = src_wav.stem

    out: list[Window] = []
    start = 0
    while start + win_len <= len(audio):
        start_time = start / sr
        label = label_for_time(regions, start_time, window_seconds=window_seconds)
        if label is not None and label != "skip":
            chunk = audio[start : start + win_len]
            out_path = out_dir / f"{source_video}_{start_time:07.2f}.wav"
            sf.write(out_path, chunk, sr)
            out.append(Window(out_path, source_video, start_time, label=label))
        start += hop
    return out
