import numpy as np
import soundfile as sf

from carzam.audio import load_wav, resample_to


def make_sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_load_wav_returns_mono_float32(tmp_path):
    sr = 16000
    wav = make_sine(440.0, sr, 1.0)
    path = tmp_path / "tone.wav"
    sf.write(path, wav, sr)
    audio, out_sr = load_wav(path)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert out_sr == sr
    assert len(audio) == sr


def test_load_wav_downmixes_stereo(tmp_path):
    sr = 16000
    left = make_sine(440.0, sr, 1.0)
    right = make_sine(880.0, sr, 1.0)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, sr)
    audio, _ = load_wav(path)
    assert audio.ndim == 1
    assert len(audio) == sr


def test_resample_changes_length():
    audio = make_sine(440.0, 16000, 1.0)
    out = resample_to(audio, src_sr=16000, dst_sr=32000)
    assert abs(len(out) - 32000) <= 2


def test_resample_noop_when_rates_match():
    audio = make_sine(440.0, 16000, 1.0)
    out = resample_to(audio, src_sr=16000, dst_sr=16000)
    assert out is audio
