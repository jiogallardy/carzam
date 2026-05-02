from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float32. Returns (audio, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), sr


def resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono float32 audio. Returns the same array if rates match."""
    if src_sr == dst_sr:
        return audio
    return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr).astype(
        np.float32, copy=False
    )
