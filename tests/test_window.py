import numpy as np
import soundfile as sf

from carai.data.window import slice_into_windows


def make_loud_signal(sr: int, duration: float, freq: float = 200.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_slices_into_correct_count(tmp_path):
    sr = 16000
    # 12.5s with 5s windows hopping 2.5s -> starts at 0, 2.5, 5.0, 7.5 (4 windows)
    audio = make_loud_signal(sr, 12.5)
    src = tmp_path / "src.wav"
    sf.write(src, audio, sr)
    out_dir = tmp_path / "windows"
    windows = slice_into_windows(
        src,
        out_dir,
        window_seconds=5.0,
        hop_seconds=2.5,
        rms_threshold=0.001,
    )
    assert len(windows) == 4
    assert {w.start_time for w in windows} == {0.0, 2.5, 5.0, 7.5}


def test_silent_windows_dropped(tmp_path):
    sr = 16000
    # 1s of loud signal then 11s of silence; only window 0 contains enough energy
    audio = np.zeros(int(sr * 12), dtype=np.float32)
    audio[: int(sr * 1)] = make_loud_signal(sr, 1.0)
    src = tmp_path / "src.wav"
    sf.write(src, audio, sr)
    out_dir = tmp_path / "windows"
    windows = slice_into_windows(
        src,
        out_dir,
        window_seconds=5.0,
        hop_seconds=2.5,
        rms_threshold=0.05,
    )
    assert len(windows) == 1
    assert windows[0].start_time == 0.0


def test_writes_wav_files(tmp_path):
    sr = 16000
    audio = make_loud_signal(sr, 6.0)
    src = tmp_path / "myvid.wav"
    sf.write(src, audio, sr)
    out_dir = tmp_path / "out"
    windows = slice_into_windows(src, out_dir, 5.0, 2.5, 0.001)
    assert len(windows) >= 1
    for w in windows:
        assert w.path.exists()
        loaded, loaded_sr = sf.read(w.path, dtype="float32")
        assert loaded_sr == sr
        assert len(loaded) == sr * 5
