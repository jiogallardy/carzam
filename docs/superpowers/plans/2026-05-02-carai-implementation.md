# carAI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch audio classifier that predicts car make and engine state (idle/accel/decel) from 3–5 second WAV clips of seven specific vehicles plus an "other" class, then run a real training cycle on YouTube-sourced data and verify on a held-out clip.

**Architecture:** Five-stage CLI pipeline (`download` → `window` → `label` → `train` → `infer`) with a pretrained PANNs CNN14 backbone driving two classification heads. Each stage reads/writes flat files (WAVs + a CSV manifest) so every stage is independently runnable and resumable.

**Tech Stack:** Python 3.11+, PyTorch with MPS, PANNs CNN14 (vendored), torchaudio, librosa, click, rich, yt-dlp, sounddevice, uv.

---

## File Structure

```
carAI/
├── pyproject.toml              # uv-managed project + deps
├── README.md
├── .gitignore                  # data/, runs/, .venv/, *.pth
├── config/
│   ├── sources.yaml            # car -> [youtube urls]
│   └── train.yaml              # hyperparams (stable defaults)
├── src/carzam/
│   ├── __init__.py
│   ├── cli.py                  # click dispatcher
│   ├── audio.py                # shared resample + load helpers
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py         # yt-dlp wrapper
│   │   ├── window.py           # 5s sliding windows + RMS gate
│   │   ├── manifest.py         # CSV I/O
│   │   ├── splits.py           # by-source-video split
│   │   ├── labeler.py          # interactive CLI labeler
│   │   └── dataset.py          # PyTorch Dataset + augs
│   └── models/
│       ├── __init__.py
│       ├── cnn14.py            # vendored PANNs CNN14
│       ├── backbone.py         # CNN14 weights loader, embedding output
│       └── multihead.py        # backbone + 2 heads
├── scripts/
│   └── fetch_panns_weights.sh
├── tests/
│   ├── test_manifest.py
│   ├── test_window.py
│   ├── test_splits.py
│   ├── test_audio.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_infer_smoke.py
├── data/                       # gitignored
└── runs/                       # gitignored
```

Each module owns one responsibility. `audio.py` is shared utility used by `download`, `window`, `dataset`, `infer` to keep resample/load consistent.

---

## Task 1: Repo scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`
- Create: `src/carzam/__init__.py`, `src/carzam/data/__init__.py`, `src/carzam/models/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
data/
runs/
*.pth
.DS_Store
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "carai"
version = "0.1.0"
description = "Car engine sound classifier"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "torchaudio>=2.3",
    "numpy>=1.26",
    "pyyaml>=6",
    "click>=8.1",
    "rich>=13",
    "yt-dlp>=2024.1.1",
    "soundfile>=0.12",
    "sounddevice>=0.4.6",
    "librosa>=0.10",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.4", "mypy>=1.10"]

[project.scripts]
carai = "carzam.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/carzam"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Create `README.md`**

```markdown
# carAI

Audio classifier that identifies car make + engine state (idle/accel/decel) from a short WAV clip.

## Setup

\`\`\`bash
uv sync
bash scripts/fetch_panns_weights.sh
\`\`\`

## Run

\`\`\`bash
carzam download
carzam window
carzam label
carzam train
carzam infer path/to/clip.wav
\`\`\`

See `docs/superpowers/specs/2026-05-01-carai-design.md` for design.
```

- [ ] **Step 4: Create empty `__init__.py` files**

`src/carzam/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/carzam/data/__init__.py`: empty file
`src/carzam/models/__init__.py`: empty file
`tests/__init__.py`: empty file

- [ ] **Step 5: Run `uv sync` to bootstrap venv**

Run: `uv sync`
Expected: creates `.venv/`, installs all deps, no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md src/carzam/__init__.py src/carzam/data/__init__.py src/carzam/models/__init__.py tests/__init__.py
git commit -m "chore: scaffold carai project layout"
```

---

## Task 2: Audio utilities (load + resample)

**Files:**
- Create: `src/carzam/audio.py`
- Test: `tests/test_audio.py`

This module is shared by every stage that touches WAVs. Centralizing avoids subtle resample mismatches.

- [ ] **Step 1: Write the failing test**

`tests/test_audio.py`:
```python
import numpy as np
import soundfile as sf
import pytest
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


def test_resample_changes_length(tmp_path):
    audio = make_sine(440.0, 16000, 1.0)
    out = resample_to(audio, src_sr=16000, dst_sr=32000)
    assert abs(len(out) - 32000) <= 2  # resampler may be off by 1-2 samples


def test_resample_noop_when_rates_match():
    audio = make_sine(440.0, 16000, 1.0)
    out = resample_to(audio, src_sr=16000, dst_sr=16000)
    assert out is audio  # same object, no copy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with ImportError on `carai.audio`.

- [ ] **Step 3: Implement `audio.py`**

`src/carzam/audio.py`:
```python
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/audio.py tests/test_audio.py
git commit -m "feat: add audio load + resample helpers"
```

---

## Task 3: Manifest CSV module

**Files:**
- Create: `src/carzam/data/manifest.py`
- Test: `tests/test_manifest.py`

The manifest is the single source of truth for what's labeled. Schema: `path, car, source_video, start_time, label`.

- [ ] **Step 1: Write the failing test**

`tests/test_manifest.py`:
```python
import pytest
from pathlib import Path
from carzam.data.manifest import (
    ManifestRow,
    write_manifest,
    read_manifest,
    append_row,
    LABELS,
)


def test_labels_constant():
    assert set(LABELS) == {"idle", "accel", "decel"}


def test_write_then_read_roundtrip(tmp_path):
    rows = [
        ManifestRow(
            path="data/windows/ferrari_812/abc_47.5.wav",
            car="ferrari_812",
            source_video="abc",
            start_time=47.5,
            label="accel",
        ),
        ManifestRow(
            path="data/windows/porsche_gt3/xyz_12.0.wav",
            car="porsche_gt3",
            source_video="xyz",
            start_time=12.0,
            label=None,
        ),
    ]
    csv_path = tmp_path / "manifest.csv"
    write_manifest(csv_path, rows)
    loaded = read_manifest(csv_path)
    assert len(loaded) == 2
    assert loaded[0].label == "accel"
    assert loaded[1].label is None
    assert loaded[1].start_time == 12.0


def test_append_row_creates_file(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    row = ManifestRow(
        path="data/windows/x.wav",
        car="other",
        source_video="x",
        start_time=0.0,
        label="idle",
    )
    append_row(csv_path, row)
    loaded = read_manifest(csv_path)
    assert len(loaded) == 1
    assert loaded[0].car == "other"


def test_append_row_extends_file(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    write_manifest(csv_path, [
        ManifestRow("a.wav", "ferrari_812", "v1", 0.0, None),
    ])
    append_row(csv_path, ManifestRow("b.wav", "ferrari_812", "v1", 5.0, "idle"))
    loaded = read_manifest(csv_path)
    assert len(loaded) == 2


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        ManifestRow("a.wav", "ferrari_812", "v1", 0.0, "bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `manifest.py`**

`src/carzam/data/manifest.py`:
```python
import csv
from dataclasses import dataclass, asdict
from pathlib import Path

LABELS = ("idle", "accel", "decel")
FIELDS = ("path", "car", "source_video", "start_time", "label")


@dataclass
class ManifestRow:
    path: str
    car: str
    source_video: str
    start_time: float
    label: str | None  # None means unlabeled

    def __post_init__(self):
        if self.label is not None and self.label not in LABELS:
            raise ValueError(
                f"label must be one of {LABELS} or None, got {self.label!r}"
            )


def write_manifest(csv_path: Path | str, rows: list[ManifestRow]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            d["label"] = "" if d["label"] is None else d["label"]
            writer.writerow(d)


def read_manifest(csv_path: Path | str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(csv_path).open() as f:
        for r in csv.DictReader(f):
            rows.append(
                ManifestRow(
                    path=r["path"],
                    car=r["car"],
                    source_video=r["source_video"],
                    start_time=float(r["start_time"]),
                    label=r["label"] if r["label"] else None,
                )
            )
    return rows


def append_row(csv_path: Path | str, row: ManifestRow) -> None:
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        d = asdict(row)
        d["label"] = "" if d["label"] is None else d["label"]
        writer.writerow(d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/data/manifest.py tests/test_manifest.py
git commit -m "feat: manifest CSV read/write"
```

---

## Task 4: Window slicer

**Files:**
- Create: `src/carzam/data/window.py`
- Test: `tests/test_window.py`

Slices a long WAV into 5-second windows with 2.5s hop, dropping silent windows.

- [ ] **Step 1: Write the failing test**

`tests/test_window.py`:
```python
import numpy as np
import soundfile as sf
from carzam.data.window import slice_into_windows


def make_loud_signal(sr: int, duration: float, freq: float = 200.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_slices_into_correct_count(tmp_path):
    sr = 16000
    audio = make_loud_signal(sr, 12.0)
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
    # 12s with 5s windows hopping 2.5s -> starts at 0, 2.5, 5.0, 7.0(last fits)
    # last window must end <= 12.0, so starts at 0.0, 2.5, 5.0, 7.0 -> 4 windows
    assert len(windows) == 4
    assert all(w.start_time in (0.0, 2.5, 5.0, 7.0) for w in windows)


def test_silent_windows_dropped(tmp_path):
    sr = 16000
    audio = np.zeros(sr * 12, dtype=np.float32)
    audio[0 : sr * 5] = make_loud_signal(sr, 5.0)
    src = tmp_path / "src.wav"
    sf.write(src, audio, sr)
    out_dir = tmp_path / "windows"
    windows = slice_into_windows(
        src,
        out_dir,
        window_seconds=5.0,
        hop_seconds=2.5,
        rms_threshold=0.01,
    )
    # only the loud first window should survive
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_window.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `window.py`**

`src/carzam/data/window.py`:
```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from carzam.audio import load_wav


@dataclass
class Window:
    path: Path
    source_video: str
    start_time: float


def slice_into_windows(
    src_wav: Path | str,
    out_dir: Path | str,
    window_seconds: float = 5.0,
    hop_seconds: float = 2.5,
    rms_threshold: float = 0.005,
) -> list[Window]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_window.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/data/window.py tests/test_window.py
git commit -m "feat: window slicer with RMS gate"
```

---

## Task 5: By-source-video splits

**Files:**
- Create: `src/carzam/data/splits.py`
- Test: `tests/test_splits.py`

Critical correctness invariant: a `source_video` must never appear in two splits.

- [ ] **Step 1: Write the failing test**

`tests/test_splits.py`:
```python
import pytest
from carzam.data.manifest import ManifestRow
from carzam.data.splits import split_by_video, SplitResult


def make_rows(car: str, video: str, n: int, label: str = "idle") -> list[ManifestRow]:
    return [
        ManifestRow(
            path=f"data/windows/{car}/{video}_{i}.wav",
            car=car,
            source_video=video,
            start_time=float(i) * 2.5,
            label=label,
        )
        for i in range(n)
    ]


def test_no_video_leaks_across_splits():
    rows = []
    for car in ["ferrari_812", "porsche_gt3"]:
        for v in range(10):
            rows.extend(make_rows(car, f"{car}_v{v}", 5))
    res = split_by_video(rows, ratios=(0.7, 0.15, 0.15), seed=0)
    train_videos = {r.source_video for r in res.train}
    val_videos = {r.source_video for r in res.val}
    test_videos = {r.source_video for r in res.test}
    assert train_videos.isdisjoint(val_videos)
    assert train_videos.isdisjoint(test_videos)
    assert val_videos.isdisjoint(test_videos)


def test_each_car_present_in_train():
    rows = []
    for car in ["ferrari_812", "porsche_gt3", "amg_c63_m177"]:
        for v in range(10):
            rows.extend(make_rows(car, f"{car}_v{v}", 5))
    res = split_by_video(rows, (0.7, 0.15, 0.15), seed=42)
    train_cars = {r.car for r in res.train}
    assert train_cars == {"ferrari_812", "porsche_gt3", "amg_c63_m177"}


def test_seed_is_deterministic():
    rows = []
    for v in range(20):
        rows.extend(make_rows("ferrari_812", f"v{v}", 3))
    res1 = split_by_video(rows, (0.7, 0.15, 0.15), seed=7)
    res2 = split_by_video(rows, (0.7, 0.15, 0.15), seed=7)
    paths1 = [r.path for r in res1.train]
    paths2 = [r.path for r in res2.train]
    assert paths1 == paths2


def test_unlabeled_rows_excluded():
    rows = make_rows("ferrari_812", "v1", 5, label="idle")
    rows.extend([
        ManifestRow("x.wav", "ferrari_812", "v2", 0.0, None),
    ])
    res = split_by_video(rows, (0.7, 0.15, 0.15), seed=0)
    all_kept = res.train + res.val + res.test
    assert all(r.label is not None for r in all_kept)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        split_by_video([], (0.6, 0.2, 0.1), seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_splits.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `splits.py`**

`src/carzam/data/splits.py`:
```python
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from carzam.data.manifest import ManifestRow


@dataclass
class SplitResult:
    train: list[ManifestRow]
    val: list[ManifestRow]
    test: list[ManifestRow]


def split_by_video(
    rows: list[ManifestRow],
    ratios: tuple[float, float, float],
    seed: int,
) -> SplitResult:
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
    rows = [r for r in rows if r.label is not None]

    by_car_video: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.source_video not in seen[r.car]:
            seen[r.car].add(r.source_video)
            by_car_video[r.car].append(r.source_video)

    rng = random.Random(seed)
    train_videos: set[str] = set()
    val_videos: set[str] = set()
    test_videos: set[str] = set()
    for car, videos in by_car_video.items():
        videos = sorted(videos)
        rng.shuffle(videos)
        n = len(videos)
        n_train = max(1, int(round(ratios[0] * n)))
        n_val = max(1, int(round(ratios[1] * n))) if n > 2 else 0
        n_train = min(n_train, n - n_val - 1) if n - n_val - 1 > 0 else n_train
        train_videos.update(videos[:n_train])
        val_videos.update(videos[n_train : n_train + n_val])
        test_videos.update(videos[n_train + n_val :])

    train, val, test = [], [], []
    for r in rows:
        if r.source_video in train_videos:
            train.append(r)
        elif r.source_video in val_videos:
            val.append(r)
        else:
            test.append(r)
    return SplitResult(train, val, test)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_splits.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/data/splits.py tests/test_splits.py
git commit -m "feat: by-video stratified split with leak guard"
```

---

## Task 6: yt-dlp download wrapper

**Files:**
- Create: `src/carzam/data/download.py`

This stage hits the network, so we do not unit-test the network call. We DO unit-test path-building / idempotency logic.

- [ ] **Step 1: Write a small targeted test**

Append to `tests/test_audio.py` (or create `tests/test_download.py`):

`tests/test_download.py`:
```python
from pathlib import Path
from carzam.data.download import target_path_for, is_already_downloaded


def test_target_path_uses_video_id(tmp_path):
    out = target_path_for(tmp_path, "ferrari_812", "https://youtube.com/watch?v=abc123XYZ_-")
    assert out.parent == tmp_path / "ferrari_812"
    assert out.name == "abc123XYZ_-.wav"


def test_target_path_handles_short_url(tmp_path):
    out = target_path_for(tmp_path, "porsche_gt3", "https://youtu.be/dQw4w9WgXcQ")
    assert out.name == "dQw4w9WgXcQ.wav"


def test_idempotency_check(tmp_path):
    car_dir = tmp_path / "ferrari_812"
    car_dir.mkdir()
    (car_dir / "abc.wav").write_bytes(b"x")
    assert is_already_downloaded(tmp_path, "ferrari_812", "https://youtu.be/abc")
    assert not is_already_downloaded(tmp_path, "ferrari_812", "https://youtu.be/zzz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `download.py`**

`src/carzam/data/download.py`:
```python
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import yaml
from rich.console import Console

console = Console()


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
    """Download audio-only as 16kHz mono WAV via yt-dlp."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_template = str(out_path.with_suffix(".%(ext)s"))
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "-o", tmp_template,
        url,
    ]
    subprocess.run(cmd, check=True)
    # yt-dlp writes <stem>.wav; rename if needed
    if not out_path.exists():
        # yt-dlp may have written a different stem; find any .wav in the dir matching the id
        candidates = list(out_path.parent.glob(f"{out_path.stem}*.wav"))
        if candidates:
            candidates[0].rename(out_path)


def download_all(sources_yaml: Path, raw_dir: Path) -> None:
    sources = yaml.safe_load(sources_yaml.read_text())
    for car, urls in sources.items():
        for url in urls:
            if is_already_downloaded(raw_dir, car, url):
                console.print(f"[dim]skip[/dim] {car} {url}")
                continue
            out_path = target_path_for(raw_dir, car, url)
            console.print(f"[bold]dl[/bold] {car} {url} -> {out_path}")
            try:
                download_one(url, out_path)
            except subprocess.CalledProcessError as e:
                console.print(f"[red]fail[/red] {url}: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/data/download.py tests/test_download.py
git commit -m "feat: yt-dlp download wrapper with idempotency"
```

---

## Task 7: Interactive labeler

**Files:**
- Create: `src/carzam/data/labeler.py`

Interactive — we don't fully unit-test the playback loop. We DO unit-test the resume logic (skip already-labeled rows).

- [ ] **Step 1: Write the targeted test**

`tests/test_labeler.py`:
```python
from pathlib import Path
from carzam.data.manifest import ManifestRow
from carzam.data.labeler import unlabeled_rows


def test_filters_to_unlabeled():
    rows = [
        ManifestRow("a.wav", "ferrari_812", "v1", 0.0, "idle"),
        ManifestRow("b.wav", "ferrari_812", "v1", 2.5, None),
        ManifestRow("c.wav", "porsche_gt3", "v2", 0.0, None),
        ManifestRow("d.wav", "porsche_gt3", "v2", 5.0, "accel"),
    ]
    out = unlabeled_rows(rows)
    assert len(out) == 2
    assert {r.path for r in out} == {"b.wav", "c.wav"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_labeler.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `labeler.py`**

`src/carzam/data/labeler.py`:
```python
import sys
import termios
import tty
from pathlib import Path

import sounddevice as sd
from rich.console import Console

from carzam.audio import load_wav
from carzam.data.manifest import (
    ManifestRow,
    LABELS,
    read_manifest,
    write_manifest,
)

console = Console()


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


KEY_TO_LABEL = {"i": "idle", "a": "accel", "d": "decel"}


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
                # leave as None; comes back next session
                break
            if key in KEY_TO_LABEL:
                by_path[row.path].label = KEY_TO_LABEL[key]
                # save after every label to avoid losing progress
                write_manifest(manifest_csv, rows)
                console.print(f"[green]✓[/green] {KEY_TO_LABEL[key]}")
                break
    write_manifest(manifest_csv, rows)
    console.print("[green]all done[/green]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_labeler.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/data/labeler.py tests/test_labeler.py
git commit -m "feat: interactive labeler with resume + per-row save"
```

---

## Task 8: Vendored PANNs CNN14

**Files:**
- Create: `src/carzam/models/cnn14.py`
- Create: `scripts/fetch_panns_weights.sh`

We vendor the CNN14 architecture from the PANNs paper (Kong et al. 2020) so we can fine-tune. Weights are downloaded once via shell script.

- [ ] **Step 1: Create weight fetch script**

`scripts/fetch_panns_weights.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
URL="https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
DEST="weights/Cnn14_mAP=0.431.pth"
if [ -f "$DEST" ]; then
  echo "weights already present at $DEST"
  exit 0
fi
echo "downloading PANNs CNN14 weights (~80MB)..."
curl -L --fail -o "$DEST" "$URL"
echo "saved to $DEST"
```

Make it executable:
```bash
chmod +x scripts/fetch_panns_weights.sh
```

- [ ] **Step 2: Vendor `cnn14.py`**

`src/carzam/models/cnn14.py`:
```python
"""Minimal PANNs CNN14 implementation, adapted from
https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT License).

We strip the AudioSet 527-class head; the model returns 2048-dim embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        for m in [self.conv1, self.conv2]:
            init_layer(m)
        for m in [self.bn1, self.bn2]:
            init_bn(m)

    def forward(self, x: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """Logmel-input CNN14. Input: (B, 1, T_frames, 64_mels).
    Output: 2048-dim embedding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        # logmel: (B, T, 64) -> (B, 1, T, 64) -> normalize across mel axis
        x = logmel.unsqueeze(1)
        x = x.transpose(1, 3)  # (B, 64, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, 64)
        x = self.conv_block1(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, (2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, (1, 1))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)  # average over mel axis
        x1 = F.max_pool1d(x, kernel_size=x.shape[-1]).flatten(1)
        x2 = F.avg_pool1d(x, kernel_size=x.shape[-1]).flatten(1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        return F.relu_(self.fc1(x))  # (B, 2048)
```

- [ ] **Step 3: Commit**

```bash
git add src/carzam/models/cnn14.py scripts/fetch_panns_weights.sh
git commit -m "feat: vendor PANNs CNN14 backbone"
```

---

## Task 9: Backbone weight loader

**Files:**
- Create: `src/carzam/models/backbone.py`
- Test: extend `tests/test_models.py`

- [ ] **Step 1: Write the targeted test**

`tests/test_models.py`:
```python
import torch
from carzam.models.backbone import build_backbone


def test_backbone_output_shape():
    model = build_backbone(weights_path=None)  # random init for test
    model.eval()
    # 5s @ 32kHz = 160000 samples; logmel ~ (B, 501, 64)
    x = torch.randn(2, 501, 64)
    with torch.no_grad():
        emb = model(x)
    assert emb.shape == (2, 2048)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_backbone_output_shape -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `backbone.py`**

`src/carzam/models/backbone.py`:
```python
from pathlib import Path
import torch
from carzam.models.cnn14 import Cnn14


def build_backbone(weights_path: Path | str | None) -> Cnn14:
    model = Cnn14()
    if weights_path is None:
        return model
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    # CNN14 checkpoint has classifier weights we don't want
    own = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing = set(own.keys()) - set(loaded.keys())
    own.update(loaded)
    model.load_state_dict(own)
    if missing:
        print(f"[backbone] missing {len(missing)} keys (heads, expected)")
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::test_backbone_output_shape -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/models/backbone.py tests/test_models.py
git commit -m "feat: backbone loader with shape-safe weight transfer"
```

---

## Task 10: Multi-head classifier

**Files:**
- Create: `src/carzam/models/multihead.py`
- Test: extend `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:
```python
from carzam.models.multihead import CarAudioModel, CARS, STATES


def test_class_constants():
    assert len(CARS) == 8
    assert "other" in CARS
    assert STATES == ("idle", "accel", "decel")


def test_multihead_forward_shapes():
    model = CarAudioModel(weights_path=None)
    model.eval()
    x = torch.randn(3, 501, 64)
    with torch.no_grad():
        car_logits, state_logits = model(x)
    assert car_logits.shape == (3, 8)
    assert state_logits.shape == (3, 3)


def test_multihead_grad_flows():
    model = CarAudioModel(weights_path=None)
    x = torch.randn(2, 501, 64)
    car_logits, state_logits = model(x)
    loss = car_logits.sum() + state_logits.sum()
    loss.backward()
    assert model.car_head.weight.grad is not None
    assert model.state_head.weight.grad is not None
    # backbone params should also have grads
    assert any(p.grad is not None for p in model.backbone.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: 3 failures (the new tests).

- [ ] **Step 3: Implement `multihead.py`**

`src/carzam/models/multihead.py`:
```python
from pathlib import Path
import torch
import torch.nn as nn
from carzam.models.backbone import build_backbone

CARS = (
    "ferrari_812",
    "lamborghini_huracan",
    "porsche_gt3",
    "amg_c63_m177",
    "bmw_m3_s58",
    "subaru_wrx_sti",
    "civic_type_r_k20c1",
    "other",
)
STATES = ("idle", "accel", "decel")


class CarAudioModel(nn.Module):
    def __init__(self, weights_path: Path | str | None) -> None:
        super().__init__()
        self.backbone = build_backbone(weights_path)
        self.car_head = nn.Linear(2048, len(CARS))
        self.state_head = nn.Linear(2048, len(STATES))
        nn.init.xavier_uniform_(self.car_head.weight)
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.zeros_(self.car_head.bias)
        nn.init.zeros_(self.state_head.bias)

    def forward(self, logmel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(logmel)
        return self.car_head(emb), self.state_head(emb)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/models/multihead.py tests/test_models.py
git commit -m "feat: multi-head model with car + state heads"
```

---

## Task 11: Dataset with augmentations

**Files:**
- Create: `src/carzam/data/dataset.py`
- Test: `tests/test_dataset.py`

The Dataset loads a window WAV, resamples 16k→32k, computes log-mel, applies augmentations during training.

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
import numpy as np
import soundfile as sf
import torch
from carzam.data.manifest import ManifestRow
from carzam.data.dataset import CarAudioDataset, compute_logmel


def make_window(tmp_path, sr=16000) -> str:
    audio = (0.3 * np.random.RandomState(0).randn(sr * 5)).astype(np.float32)
    p = tmp_path / "x.wav"
    sf.write(p, audio, sr)
    return str(p)


def test_logmel_shape():
    audio = torch.randn(32000 * 5)  # 5s @ 32kHz
    mel = compute_logmel(audio, sample_rate=32000)
    # expect (frames, 64); frames depend on hop
    assert mel.dim() == 2
    assert mel.shape[1] == 64
    assert mel.shape[0] > 100


def test_dataset_returns_correct_items(tmp_path):
    path = make_window(tmp_path)
    rows = [
        ManifestRow(path, "ferrari_812", "v1", 0.0, "accel"),
        ManifestRow(path, "porsche_gt3", "v2", 2.5, "idle"),
    ]
    ds = CarAudioDataset(rows, train=False)
    item = ds[0]
    assert "logmel" in item
    assert "car_idx" in item
    assert "state_idx" in item
    assert item["logmel"].shape[1] == 64
    assert item["car_idx"] == 0  # ferrari_812 is index 0
    assert item["state_idx"] == 1  # accel is index 1


def test_train_augmentations_change_output(tmp_path):
    path = make_window(tmp_path)
    rows = [ManifestRow(path, "ferrari_812", "v1", 0.0, "accel")]
    ds = CarAudioDataset(rows, train=True, seed=0)
    a = ds[0]["logmel"]
    ds = CarAudioDataset(rows, train=True, seed=1)
    b = ds[0]["logmel"]
    # different seeds -> different augmented output
    assert not torch.equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `dataset.py`**

`src/carzam/data/dataset.py`:
```python
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

from carzam.audio import load_wav, resample_to
from carzam.data.manifest import ManifestRow
from carzam.models.multihead import CARS, STATES

SAMPLE_RATE = 32000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320  # 32000 * 5 / 320 ~= 500 frames per 5s window


def compute_logmel(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Returns log-mel of shape (frames, n_mels)."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=20.0,
        f_max=sample_rate / 2,
    )(audio)
    log_mel = torch.log(mel + 1e-6)
    return log_mel.squeeze(0).transpose(0, 1)  # (frames, n_mels)


class CarAudioDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        train: bool,
        seed: int = 0,
    ) -> None:
        self.rows = [r for r in rows if r.label is not None]
        self.train = train
        self.rng = random.Random(seed)
        # SpecAugment masks
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=40)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        audio, sr = load_wav(row.path)
        if sr != SAMPLE_RATE:
            audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
        audio_t = torch.from_numpy(np.ascontiguousarray(audio))

        if self.train:
            # random gain ±6dB
            gain_db = (self.rng.random() * 12.0) - 6.0
            audio_t = audio_t * (10.0 ** (gain_db / 20.0))

        logmel = compute_logmel(audio_t)

        if self.train:
            logmel = logmel.transpose(0, 1).unsqueeze(0)  # (1, mel, frames)
            logmel = self.freq_mask(logmel)
            logmel = self.time_mask(logmel)
            logmel = logmel.squeeze(0).transpose(0, 1)

        return {
            "logmel": logmel,
            "car_idx": CARS.index(row.car),
            "state_idx": STATES.index(row.label),  # type: ignore[arg-type]
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/data/dataset.py tests/test_dataset.py
git commit -m "feat: dataset with logmel + SpecAugment + gain aug"
```

---

## Task 12: Training config + train.yaml

**Files:**
- Create: `config/train.yaml`

Stable defaults captured in YAML so re-running uses the same hyperparams.

- [ ] **Step 1: Create `config/train.yaml`**

```yaml
# carai training defaults

paths:
  manifest: data/manifest.csv
  weights: weights/Cnn14_mAP=0.431.pth
  runs_dir: runs

splits:
  ratios: [0.7, 0.15, 0.15]
  seed: 42

train:
  batch_size: 32
  num_workers: 0          # MPS-safe
  max_epochs: 50
  early_stop_patience: 5
  lr_head: 1.0e-4
  lr_backbone: 1.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 2
  car_loss_weight: 1.0
  state_loss_weight: 1.0
```

- [ ] **Step 2: Commit**

```bash
git add config/train.yaml
git commit -m "chore: add training config defaults"
```

---

## Task 13: Training loop

**Files:**
- Create: `src/carzam/train.py`

No unit test for the loop itself — instead a smoke run on a tiny synthetic dataset is part of Task 17.

- [ ] **Step 1: Implement `train.py`**

`src/carzam/train.py`:
```python
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from rich.console import Console
from torch.utils.data import DataLoader

from carzam.data.dataset import CarAudioDataset
from carzam.data.manifest import read_manifest
from carzam.data.splits import split_by_video
from carzam.models.multihead import CarAudioModel

console = Console()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    console.print("[yellow]no GPU found, using CPU[/yellow]")
    return torch.device("cpu")


def collate(batch: list[dict]) -> dict:
    # pad logmels along time axis to the longest in the batch
    mels = [b["logmel"] for b in batch]
    max_t = max(m.shape[0] for m in mels)
    n_mel = mels[0].shape[1]
    out = torch.zeros(len(mels), max_t, n_mel, dtype=torch.float32)
    for i, m in enumerate(mels):
        out[i, : m.shape[0]] = m
    return {
        "logmel": out,
        "car_idx": torch.tensor([b["car_idx"] for b in batch], dtype=torch.long),
        "state_idx": torch.tensor([b["state_idx"] for b in batch], dtype=torch.long),
    }


def make_optimizer(model: CarAudioModel, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
            {"params": model.car_head.parameters(), "lr": cfg["train"]["lr_head"]},
            {"params": model.state_head.parameters(), "lr": cfg["train"]["lr_head"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )


def cosine_lr(epoch: int, total: int, warmup: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class EpochStats:
    car_loss: float
    state_loss: float
    car_acc: float
    state_acc: float


def run_epoch(
    model: CarAudioModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)
    car_loss_sum = state_loss_sum = 0.0
    car_correct = state_correct = total = 0
    for batch in loader:
        logmel = batch["logmel"].to(device)
        car_y = batch["car_idx"].to(device)
        state_y = batch["state_idx"].to(device)
        with torch.set_grad_enabled(is_train):
            car_logits, state_logits = model(logmel)
            car_loss = F.cross_entropy(car_logits, car_y)
            state_loss = F.cross_entropy(state_logits, state_y)
            loss = (
                cfg["train"]["car_loss_weight"] * car_loss
                + cfg["train"]["state_loss_weight"] * state_loss
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        car_loss_sum += car_loss.item() * logmel.size(0)
        state_loss_sum += state_loss.item() * logmel.size(0)
        car_correct += (car_logits.argmax(1) == car_y).sum().item()
        state_correct += (state_logits.argmax(1) == state_y).sum().item()
        total += logmel.size(0)
    return EpochStats(
        car_loss=car_loss_sum / total,
        state_loss=state_loss_sum / total,
        car_acc=car_correct / total,
        state_acc=state_correct / total,
    )


def train(config_path: Path) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    torch.set_default_dtype(torch.float32)

    rows = read_manifest(cfg["paths"]["manifest"])
    splits = split_by_video(
        rows,
        ratios=tuple(cfg["splits"]["ratios"]),
        seed=cfg["splits"]["seed"],
    )
    console.print(
        f"split: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}"
    )

    train_ds = CarAudioDataset(splits.train, train=True, seed=cfg["splits"]["seed"])
    val_ds = CarAudioDataset(splits.val, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
    )

    device = pick_device()
    weights_path = Path(cfg["paths"]["weights"])
    model = CarAudioModel(weights_path if weights_path.exists() else None).to(device)
    optimizer = make_optimizer(model, cfg)

    run_dir = Path(cfg["paths"]["runs_dir"]) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    best_val = -1.0
    patience = 0
    history: list[dict] = []
    for epoch in range(cfg["train"]["max_epochs"]):
        scale = cosine_lr(epoch, cfg["train"]["max_epochs"], cfg["train"]["warmup_epochs"])
        for group in optimizer.param_groups:
            group["lr"] = group["lr"] if epoch == 0 else group["lr"]
        # apply schedule by scaling base LRs each epoch
        base_lrs = [cfg["train"]["lr_backbone"], cfg["train"]["lr_head"], cfg["train"]["lr_head"]]
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

        ts = time.time()
        tr = run_epoch(model, train_loader, optimizer, device, cfg)
        va = run_epoch(model, val_loader, None, device, cfg)
        elapsed = time.time() - ts
        console.print(
            f"epoch {epoch:02d}  "
            f"train: car_loss={tr.car_loss:.3f} car_acc={tr.car_acc:.3f} "
            f"state_acc={tr.state_acc:.3f}  "
            f"val: car_acc={va.car_acc:.3f} state_acc={va.state_acc:.3f}  "
            f"({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch, "train": tr.__dict__, "val": va.__dict__})
        if va.car_acc > best_val:
            best_val = va.car_acc
            patience = 0
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
            console.print(f"  [green]saved checkpoint (val car_acc={best_val:.3f})[/green]")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                console.print(f"[yellow]early stop at epoch {epoch}[/yellow]")
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    console.print(f"[bold]done.[/bold] run dir: {run_dir}")
    return run_dir
```

- [ ] **Step 2: Commit**

```bash
git add src/carzam/train.py
git commit -m "feat: training loop with cosine LR, early stop, MPS support"
```

---

## Task 14: Eval module with confusion matrices

**Files:**
- Create: `src/carzam/eval.py`

- [ ] **Step 1: Implement `eval.py`**

`src/carzam/eval.py`:
```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich.console import Console
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from carzam.data.dataset import CarAudioDataset
from carzam.data.manifest import read_manifest
from carzam.data.splits import split_by_video
from carzam.models.multihead import CarAudioModel, CARS, STATES
from carzam.train import collate, pick_device

console = Console()


def _plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(6, len(labels))))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate(run_dir: Path, split: str = "test") -> None:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    rows = read_manifest(cfg["paths"]["manifest"])
    splits = split_by_video(
        rows,
        ratios=tuple(cfg["splits"]["ratios"]),
        seed=cfg["splits"]["seed"],
    )
    chosen = {"train": splits.train, "val": splits.val, "test": splits.test}[split]
    ds = CarAudioDataset(chosen, train=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)

    device = pick_device()
    model = CarAudioModel(weights_path=None).to(device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device))
    model.eval()

    car_true, car_pred, state_true, state_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logmel = batch["logmel"].to(device)
            cl, sl = model(logmel)
            car_pred.extend(cl.argmax(1).cpu().tolist())
            state_pred.extend(sl.argmax(1).cpu().tolist())
            car_true.extend(batch["car_idx"].tolist())
            state_true.extend(batch["state_idx"].tolist())

    car_report = classification_report(
        car_true, car_pred, target_names=list(CARS), output_dict=True, zero_division=0
    )
    state_report = classification_report(
        state_true, state_pred, target_names=list(STATES), output_dict=True, zero_division=0
    )
    metrics = {"car": car_report, "state": state_report, "split": split}
    (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))

    cm_car = confusion_matrix(car_true, car_pred, labels=list(range(len(CARS))))
    cm_state = confusion_matrix(state_true, state_pred, labels=list(range(len(STATES))))
    _plot_confusion(cm_car, list(CARS), run_dir / f"confusion_car_{split}.png")
    _plot_confusion(cm_state, list(STATES), run_dir / f"confusion_state_{split}.png")

    console.print(
        f"[bold]{split}[/bold]  "
        f"car_acc={car_report['accuracy']:.3f}  "
        f"state_acc={state_report['accuracy']:.3f}"
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/carzam/eval.py
git commit -m "feat: eval module with classification reports + confusion plots"
```

---

## Task 15: Inference module

**Files:**
- Create: `src/carzam/infer.py`
- Test: `tests/test_infer_smoke.py`

- [ ] **Step 1: Write the smoke test**

`tests/test_infer_smoke.py`:
```python
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

from carzam.infer import predict_clip
from carzam.models.multihead import CarAudioModel


def test_predict_clip_returns_expected_shape(tmp_path):
    sr = 16000
    audio = (0.3 * np.random.RandomState(0).randn(sr * 5)).astype(np.float32)
    p = tmp_path / "clip.wav"
    sf.write(p, audio, sr)

    model = CarAudioModel(weights_path=None)
    model.eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt)

    result = predict_clip(p, ckpt, device=torch.device("cpu"))
    assert "car" in result and "state" in result
    assert "car_confidence" in result
    assert 0.0 <= result["car_confidence"] <= 1.0
    assert "car_top3" in result and len(result["car_top3"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_infer_smoke.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `infer.py`**

`src/carzam/infer.py`:
```python
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from carzam.audio import load_wav, resample_to
from carzam.data.dataset import compute_logmel, SAMPLE_RATE
from carzam.models.multihead import CarAudioModel, CARS, STATES

WINDOW_SECONDS = 5.0
OTHER_THRESHOLD = 0.40  # below this, force "other"


def _prepare_audio(path: Path) -> torch.Tensor:
    audio, sr = load_wav(path)
    audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
    target_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(audio) < int(2.0 * SAMPLE_RATE):
        raise ValueError(f"clip too short: {len(audio) / SAMPLE_RATE:.1f}s, need >= 2s")
    if len(audio) < target_len:
        pad = np.zeros(target_len - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    elif len(audio) > target_len:
        # center crop
        excess = len(audio) - target_len
        start = excess // 2
        audio = audio[start : start + target_len]
    return torch.from_numpy(np.ascontiguousarray(audio))


def _is_url(path_or_url: str | Path) -> bool:
    s = str(path_or_url)
    return s.startswith("http://") or s.startswith("https://")


def _download_url_to_wav(url: str, out_dir: Path) -> Path:
    out = out_dir / "clip.wav"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "-o", str(out.with_suffix(".%(ext)s")),
        url,
    ]
    subprocess.run(cmd, check=True)
    if not out.exists():
        cands = list(out_dir.glob("*.wav"))
        if cands:
            cands[0].rename(out)
    return out


def predict_clip(
    path_or_url: str | Path,
    checkpoint: Path,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cpu")
    if _is_url(path_or_url):
        with tempfile.TemporaryDirectory() as td:
            wav = _download_url_to_wav(str(path_or_url), Path(td))
            return predict_clip(wav, checkpoint, device)

    audio = _prepare_audio(Path(path_or_url))
    logmel = compute_logmel(audio).unsqueeze(0).to(device)

    model = CarAudioModel(weights_path=None).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        car_logits, state_logits = model(logmel)
        car_probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()
        state_probs = F.softmax(state_logits, dim=-1).squeeze(0).cpu().numpy()

    car_top1 = int(np.argmax(car_probs))
    state_top1 = int(np.argmax(state_probs))
    car_conf = float(car_probs[car_top1])
    car_label = CARS[car_top1] if car_conf >= OTHER_THRESHOLD else "other"

    car_order = np.argsort(-car_probs)
    state_order = np.argsort(-state_probs)
    return {
        "car": car_label,
        "car_confidence": car_conf,
        "state": STATES[state_top1],
        "state_confidence": float(state_probs[state_top1]),
        "car_top3": [(CARS[i], float(car_probs[i])) for i in car_order[:3]],
        "state_all": [(STATES[i], float(state_probs[i])) for i in state_order],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_infer_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/carzam/infer.py tests/test_infer_smoke.py
git commit -m "feat: inference module with URL and file inputs"
```

---

## Task 16: CLI dispatcher

**Files:**
- Create: `src/carzam/cli.py`

- [ ] **Step 1: Implement `cli.py`**

`src/carzam/cli.py`:
```python
from pathlib import Path
import json
import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def main() -> None:
    """carai — car engine sound classifier"""


@main.command()
@click.option("--sources", type=click.Path(path_type=Path), default=Path("config/sources.yaml"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/raw"))
def download(sources: Path, out: Path) -> None:
    """Download videos listed in config/sources.yaml as 16kHz mono WAVs."""
    from carzam.data.download import download_all
    download_all(sources, out)


@main.command()
@click.option("--raw", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/windows"))
@click.option("--manifest", type=click.Path(path_type=Path), default=Path("data/manifest.csv"))
@click.option("--rms-threshold", type=float, default=0.005)
def window(raw: Path, out: Path, manifest: Path, rms_threshold: float) -> None:
    """Slice raw WAVs into 5-second windows and write a pre-label manifest."""
    from carzam.data.window import slice_into_windows
    from carzam.data.manifest import ManifestRow, write_manifest

    rows: list[ManifestRow] = []
    for car_dir in sorted(raw.iterdir()):
        if not car_dir.is_dir():
            continue
        car = car_dir.name
        car_out = out / car
        for src in sorted(car_dir.glob("*.wav")):
            console.print(f"[dim]window[/dim] {src}")
            wins = slice_into_windows(src, car_out, 5.0, 2.5, rms_threshold)
            for w in wins:
                rows.append(
                    ManifestRow(
                        path=str(w.path),
                        car=car,
                        source_video=w.source_video,
                        start_time=w.start_time,
                        label=None,
                    )
                )
    write_manifest(manifest, rows)
    console.print(f"[green]wrote {len(rows)} windows to {manifest}[/green]")


@main.command()
@click.option("--manifest", type=click.Path(path_type=Path), default=Path("data/manifest.csv"))
def label(manifest: Path) -> None:
    """Interactive labeler. Resumes where you left off."""
    from carzam.data.labeler import label_session
    label_session(manifest)


@main.command()
@click.option("--config", type=click.Path(path_type=Path), default=Path("config/train.yaml"))
def train(config: Path) -> None:
    """Train the model."""
    from carzam.train import train as train_run
    train_run(config)


@main.command()
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--split", type=click.Choice(["train", "val", "test"]), default="test")
def eval_(run_dir: Path, split: str) -> None:
    """Evaluate a trained run on the given split."""
    from carzam.eval import evaluate
    evaluate(run_dir, split)


# expose as `carzam eval` not `carzam eval-`
main.add_command(eval_, name="eval")


@main.command()
@click.argument("clip")
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True)
@click.option("--json", "as_json", is_flag=True)
def infer(clip: str, checkpoint: Path, as_json: bool) -> None:
    """Predict car + state for a clip (file path or YouTube URL)."""
    import torch
    from carzam.infer import predict_clip
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    result = predict_clip(clip, checkpoint, device=device)
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    table = Table(show_header=False)
    table.add_row("clip", str(clip))
    table.add_row("car", f"{result['car']}  ({result['car_confidence']:.2f})")
    for name, p in result["car_top3"]:
        table.add_row("", f"  {name}: {p:.2f}")
    table.add_row("state", f"{result['state']}  ({result['state_confidence']:.2f})")
    console.print(table)
```

- [ ] **Step 2: Smoke-check the CLI loads**

Run: `uv run carai --help`
Expected: prints commands `download, window, label, train, eval, infer`.

- [ ] **Step 3: Commit**

```bash
git add src/carzam/cli.py
git commit -m "feat: CLI dispatcher wiring all subcommands"
```

---

## Task 17: End-to-end smoke test on synthetic data

**Files:**
- Create: `tests/test_e2e_smoke.py`

Verifies the full pipeline runs without crashing on a tiny synthetic dataset. NOT testing accuracy — just shape/wire correctness.

- [ ] **Step 1: Write the smoke test**

`tests/test_e2e_smoke.py`:
```python
import numpy as np
import soundfile as sf
import torch
import yaml
from pathlib import Path

from carzam.data.manifest import ManifestRow, write_manifest
from carzam.train import train as train_run
from carzam.infer import predict_clip
from carzam.models.multihead import CARS


def make_clip(path: Path, sr: int = 16000, freq: float = 200.0) -> None:
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, x, sr)


def test_train_then_infer(tmp_path):
    # build a tiny manifest with 2 cars, 2 videos each, 4 windows each
    win_dir = tmp_path / "windows"
    win_dir.mkdir()
    rows = []
    for ci, car in enumerate(CARS[:2]):
        for v in range(2):
            for i in range(4):
                p = win_dir / f"{car}_v{v}_{i}.wav"
                make_clip(p, freq=200.0 + 50 * ci + 5 * i)
                rows.append(
                    ManifestRow(
                        path=str(p),
                        car=car,
                        source_video=f"{car}_v{v}",
                        start_time=float(i) * 2.5,
                        label="idle" if i % 2 == 0 else "accel",
                    )
                )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    cfg = {
        "paths": {
            "manifest": str(manifest),
            "weights": "weights/does_not_exist.pth",
            "runs_dir": str(tmp_path / "runs"),
        },
        "splits": {"ratios": [0.6, 0.2, 0.2], "seed": 0},
        "train": {
            "batch_size": 4,
            "num_workers": 0,
            "max_epochs": 2,
            "early_stop_patience": 5,
            "lr_head": 1e-3,
            "lr_backbone": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "car_loss_weight": 1.0,
            "state_loss_weight": 1.0,
        },
    }
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = train_run(cfg_path)
    assert (run_dir / "checkpoint.pt").exists()

    # infer on a synthetic clip (any output is fine; we test the pipeline runs)
    clip = tmp_path / "mystery.wav"
    make_clip(clip)
    result = predict_clip(clip, run_dir / "checkpoint.pt", device=torch.device("cpu"))
    assert result["car"] in list(CARS) + ["other"]
    assert result["state"] in ("idle", "accel", "decel")
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/test_e2e_smoke.py -v -s`
Expected: PASS (takes ~30–60s on M1 Max).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke from manifest to inference"
```

---

## Task 18: Populate `config/sources.yaml` (Claude does the YouTube research)

**Files:**
- Create: `config/sources.yaml`

The agent executing this task searches YouTube and produces a candidate `sources.yaml`, then asks the user to prune.

- [ ] **Step 1: Search YouTube for each car**

Use WebSearch + `yt-dlp ytsearch20:"<query>" --print "%(id)s %(channel)s %(duration)s %(title)s" --skip-download` for each query below. Limit to videos 2–12 minutes, view count ≥ 5k.

Search queries per car:
- `ferrari_812`: "Ferrari 812 Superfast pure sound", "Ferrari 812 acceleration flyby", "Ferrari 812 cold start"
- `lamborghini_huracan`: "Lamborghini Huracan pure sound", "Huracan Performante exhaust", "Huracan acceleration"
- `porsche_gt3`: "Porsche 911 GT3 sound", "992 GT3 exhaust", "GT3 RS pure sound"
- `amg_c63_m177`: "AMG C63 S sound", "C63S exhaust", "C63 acceleration"
- `bmw_m3_s58`: "BMW M3 G80 sound", "S58 exhaust", "M4 G82 acceleration"
- `subaru_wrx_sti`: "Subaru WRX STI sound", "STI EJ257 boxer rumble", "STI cold start"
- `civic_type_r_k20c1`: "Civic Type R FK8 sound", "FL5 Type R exhaust", "K20C1 acceleration"
- `other`: "Tesla Model 3 sound", "Toyota Camry idle", "Ford F-150 V8 exhaust", "Yamaha R1 acceleration", "Cummins diesel idle"

Trusted channels (boost in ranking): Marchettino, AutoTopNL, Shmee150, Mr.JWW, DailyDrivenExotics, Carfection.

- [ ] **Step 2: Write `config/sources.yaml` with ~10 URLs per car**

Format:
```yaml
ferrari_812:
  - https://www.youtube.com/watch?v=<id>  # [Channel] Title (M:SS)
  - ...
```

- [ ] **Step 3: Show the file to the user and ask for prunes**

Print the file. Ask: "Here are the candidate videos. Want to remove any obvious bad picks before we download?"

- [ ] **Step 4: Commit after user approves**

```bash
git add config/sources.yaml
git commit -m "data: seed sources.yaml with curated YouTube videos"
```

---

## Task 19: Real data run — download

**Files:** none (data only)

This is a runtime task. Pause for user confirmation before continuing — downloads may be slow.

- [ ] **Step 1: Fetch PANNs weights**

Run: `bash scripts/fetch_panns_weights.sh`
Expected: `weights/Cnn14_mAP=0.431.pth` exists, ~80MB.

- [ ] **Step 2: Run download**

Run: `uv run carzam download`
Expected: `data/raw/<car>/*.wav` populated. May take 20–60 min depending on network.

- [ ] **Step 3: Verify counts**

Run: `find data/raw -name "*.wav" | awk -F/ '{print $3}' | sort | uniq -c`
Expected: each car has 5–10 files. If any car has 0, revisit `sources.yaml`.

---

## Task 20: Real data run — windowing

- [ ] **Step 1: Run windowing**

Run: `uv run carzam window`
Expected: prints per-source progress, writes `data/windows/<car>/*.wav` and `data/manifest.csv`. Total windows ~3,000–6,000 (most will be skipped during labeling).

- [ ] **Step 2: Sanity-check the manifest**

Run: `head data/manifest.csv && wc -l data/manifest.csv`
Expected: header + thousands of rows, all with `label` column empty.

---

## Task 21: Labeling session (USER WORK — ~3–5 hours)

This task is the user's, not the agent's. The agent should explain how to run it and stop.

- [ ] **Step 1: Tell user how to start**

Message to user:
> "Time to label. Run `uv run carzam label`. You'll hear a 5-second window and press `i`/`a`/`d` for idle/accel/decel, `s` to skip junk (voiceover, music intro, silence). Quit with `q` any time — progress is saved per label. Aim for ~50–80 labels per (car, state) cell, roughly 1,200–1,700 total. Plan for 3–5 hours across multiple sessions. When done, tell me and I'll start training."

- [ ] **Step 2: Wait for user to confirm completion**

Do not proceed to Task 22 until user says labeling is done.

- [ ] **Step 3: Verify label distribution**

Run:
```bash
uv run python -c "
from carzam.data.manifest import read_manifest
from collections import Counter
rows = read_manifest('data/manifest.csv')
labeled = [(r.car, r.label) for r in rows if r.label]
print(f'total labeled: {len(labeled)}')
for k, v in sorted(Counter(labeled).items()):
    print(f'  {k}: {v}')
"
```
Expected: each (car, label) cell has at least 30. If any cell is below 20, ask user to do another short labeling pass on that car/state.

---

## Task 22: Train the model

- [ ] **Step 1: Run training**

Run: `uv run carzam train`
Expected: prints epoch logs, saves checkpoint to `runs/<timestamp>/checkpoint.pt`. Training: 20–40 minutes on M1 Max.

- [ ] **Step 2: Note the run dir**

Capture the path printed at the end (e.g. `runs/20260502_143015`). It'll be needed for eval and infer.

- [ ] **Step 3: Run eval on test split**

Run: `uv run carzam eval --run-dir runs/<timestamp> --split test`
Expected: prints `car_acc` and `state_acc`, writes `metrics_test.json` and two confusion PNGs.

- [ ] **Step 4: Inspect confusion matrices**

Open `runs/<timestamp>/confusion_car_test.png` and `confusion_state_test.png`. Look for:
- Strong diagonal on car (good).
- Off-diagonal clusters between turbo cars (AMG/BMW or Subaru/Civic) — expected, not a bug.
- State confusion mostly between accel and decel — expected.

If car_acc < 0.6 or state_acc < 0.7, something is wrong (likely too few labels or label noise). Report and stop.

---

## Task 23: Inference on a fresh clip

- [ ] **Step 1: Pick a test clip**

Either:
- Find a YouTube video the model has NOT seen (different channel, different upload), copy its URL.
- Or trim a 5s WAV from any source.

- [ ] **Step 2: Run inference**

Run: `uv run carzam infer "<url-or-path>" --checkpoint runs/<timestamp>/checkpoint.pt`
Expected: pretty table with `car`, `car_top3`, `state` and confidences.

- [ ] **Step 3: Try a deliberate "other" check**

Pass a clip of something not in the training set (e.g., a chainsaw video, a garbage truck). Expect `car: other` due to the <0.4 confidence floor.

- [ ] **Step 4: Commit run artifacts**

```bash
git add runs/<timestamp>/metrics_test.json runs/<timestamp>/confusion_car_test.png runs/<timestamp>/confusion_state_test.png runs/<timestamp>/config.yaml runs/<timestamp>/history.json
git commit -m "data: first training run results"
```

(Note: the `.gitignore` excludes `runs/`, so use `git add -f` if you want to keep these in version control.)

---

## Self-Review (post-write checklist)

**Spec coverage:**
- 8-class car taxonomy → Task 10 (CARS constant), Task 18 (sources)
- 3-state engine → Task 10 (STATES), Task 7 (labeler key map), Task 11 (dataset)
- "other" class with <0.4 confidence floor → Task 15 (`OTHER_THRESHOLD`)
- 5s window, 2.5s hop, RMS gate → Task 4
- Storage 16kHz, runtime 32kHz → Task 6 (download flag), Task 11 (dataset resample)
- PANNs CNN14 backbone → Tasks 8, 9
- Multi-head with `car_head` + `state_head`, summed cross-entropy → Tasks 10, 13
- SpecAugment + random gain, NOT pitch/time stretch → Task 11
- By-source-video splits with leak-guard test → Task 5
- MPS gotchas (fp32, num_workers=0, CPU fallback) → Tasks 12, 13
- CLI subcommands (download/window/label/train/eval/infer) → Task 16
- Inference on file or URL → Task 15
- `--json` flag → Task 16
- Confusion matrices + classification reports → Task 14
- Resumable labeler with per-row save → Task 7
- Skipping silent windows → Task 4
- Real-data end-to-end run → Tasks 18–23

No spec section is uncovered.

**Placeholder scan:** No `TBD`, `TODO`, "fill in later", or `# implement here`. Every code block is real and runnable.

**Type consistency:**
- `ManifestRow` fields used consistently across Tasks 3, 5, 7, 11, 17.
- `CarAudioModel(weights_path=...)` signature used consistently in Tasks 9, 10, 13, 15, 17.
- `CARS` and `STATES` constants imported consistently across Tasks 10, 11, 14, 15.
- `compute_logmel` shape contract `(frames, n_mels)` matches dataset → train → infer.
