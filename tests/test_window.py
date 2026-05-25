import numpy as np
import soundfile as sf

from carzam.data.regions import Region
from carzam.data.window import slice_into_windows, slice_with_regions


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


def test_slice_with_regions_inherits_label(tmp_path):
    sr = 16000
    audio = make_loud_signal(sr, 30.0)
    src = tmp_path / "src.wav"
    sf.write(src, audio, sr)
    regions = [
        Region(0.0, 5.0, "skip"),
        Region(5.0, 20.0, "accel"),
        Region(20.0, 30.0, "idle"),
    ]
    out_dir = tmp_path / "out"
    windows = slice_with_regions(src, out_dir, regions, 5.0, 2.5)
    # Window starts at 0 (skip), 2.5 (skip/accel boundary), 5 (accel), 7.5 (accel),
    # 10 (accel), 12.5 (accel), 15 (accel/end-of-accel boundary -> straddles), 17.5 (straddles),
    # 20 (idle), 22.5 (idle), 25 (idle).
    # Skip windows are dropped. Accel: windows fully inside [5, 20] are 5, 7.5, 10, 12.5
    # (each ends at 10, 12.5, 15, 17.5 -- 17.5 ends at 22.5 which exceeds 20, so 12.5 ends at 17.5 ok)
    # Wait: accel range is 5..20, win_seconds=5. start in [5, 15] qualifies. So 5, 7.5, 10, 12.5, 15.
    # But hops are 2.5 from 0: 0,2.5,5,7.5,10,12.5,15,17.5. Of these, accel-qualifying: 5,7.5,10,12.5.
    # 15 ends at 20 which is the boundary; depending on label_for_time semantics it may qualify.
    # Idle [20, 30]: starts in [20, 25]. From hop sequence: 20, 22.5, 25.
    accel_starts = sorted(w.start_time for w in windows if w.label == "accel")
    idle_starts = sorted(w.start_time for w in windows if w.label == "idle")
    skip_count = sum(1 for w in windows if w.label == "skip")
    assert skip_count == 0
    assert all(s in (5.0, 7.5, 10.0, 12.5, 15.0) for s in accel_starts)
    assert all(s in (20.0, 22.5, 25.0) for s in idle_starts)
    assert len(windows) >= 7  # at least these labelled windows


def test_slice_with_regions_handles_gaps(tmp_path):
    sr = 16000
    audio = make_loud_signal(sr, 30.0)
    src = tmp_path / "src.wav"
    sf.write(src, audio, sr)
    # Gap from 15 to 25 with no region; only 0-15 and 25-30 are labeled
    regions = [
        Region(0.0, 15.0, "accel"),
        Region(25.0, 30.0, "idle"),
    ]
    windows = slice_with_regions(src, tmp_path / "out", regions, 5.0, 2.5)
    # accel windows fit in [0,15] starting at 0, 2.5, 5, 7.5, 10
    # idle windows fit in [25,30] starting at 25
    accel_count = sum(1 for w in windows if w.label == "accel")
    idle_count = sum(1 for w in windows if w.label == "idle")
    assert accel_count == 5
    assert idle_count == 1
