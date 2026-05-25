# carzam

> Shazam, but for car engines. Identify the make/model of a car from a short clip of its exhaust note.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Built with PyTorch](https://img.shields.io/badge/built%20with-PyTorch-ee4c2c.svg)](https://pytorch.org/)

`carzam` is an end-to-end pipeline for training and running an audio classifier that recognizes cars by sound, all from one CLI. It covers:

- **Data collection** — scrape and normalize audio from a curated list of YouTube videos.
- **Labeling** — both interactive (a TUI reviewer) and automated (visual verification with YOLOv8 + DINOv2, audio screening with silero-VAD + LAION-CLAP).
- **Training** — a contrastive embedding model over PANNs CNN14, plus optional per-model specialist heads.
- **Inference** — single-clip prediction, Shazam-style 10s rolling recognition, and live-mic mode.

Current corpus: **~71 classes** spanning supercars, hypercars, JDM legends and modern flagships (Bugatti Chiron, Pagani Huayra, Lexus LFA, Porsche 992 GT3 RS, Nissan Skyline R34, …).

---

## Table of contents

- [Quickstart](#quickstart)
- [Repo layout](#repo-layout)
- [The full pipeline](#the-full-pipeline)
- [Auto-labeling](#auto-labeling)
- [Dataset](#dataset)
- [Development](#development)
- [License](#license)

---

## Quickstart

```bash
git clone https://github.com/jiogallardy/carzam.git
cd carzam

uv sync --extra train --extra auto-label
bash scripts/fetch_panns_weights.sh

# Predict on a local clip with a trained checkpoint
carzam infer path/to/clip.wav --checkpoint runs/<ts>/checkpoint.pt

# Or pipe a YouTube URL straight through
carzam infer "https://www.youtube.com/watch?v=..." --checkpoint runs/<ts>/checkpoint.pt
```

> No checkpoint yet? See [The full pipeline](#the-full-pipeline) to build one.

---

## Repo layout

```
carzam/
├── src/carzam/              # Core library + CLI
│   ├── cli.py               # `carzam` entry point (Click)
│   ├── audio.py             # WAV I/O, resampling, log-mel
│   ├── data/                # download, regions, windowing, manifest, auto-label
│   ├── models/              # CNN14 backbone + multi-head & contrastive heads
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── embedding.py         # open-set nearest-prototype matching
│   └── cascade.py
├── config/                  # Training configs + source video lists
├── scripts/                 # One-off utilities (yt classifier, audits, demos)
├── data/                    # See `Dataset` section — most is gitignored
└── tests/                   # pytest suite
```

---

## The full pipeline

### 1. Download

`config/sources.yaml` lists every (car, YouTube URL) pair. `carzam download` pulls them as 16kHz mono WAVs into `data/raw/<car>/<video_id>.wav`. By default it only pulls videos that have been approved in review (have a regions YAML); use `--all-sources` to grab everything.

```bash
carzam download
```

### 2. Label

Two paths.

**Interactive review** — play through a video and mark engine-state ranges:

```bash
carzam review
```

| key | action |
|-----|--------|
| `i` | start `idle` range |
| `a` | start `accel` range |
| `d` | start `decel` range |
| `s` | start `skip` range (intro / voiceover / music) |
| `x` | close current range |
| `space` | pause / resume |
| `z` / `c` | rewind / forward 5s |
| `u` | undo last range |
| `q` | save & quit |
| `n` | next video without saving |

**Auto-label** — let the [auto-labeler](#auto-labeling) do it.

Either way you end up with `data/regions/<car>/<video_id>.yaml` files (small text — committed to git).

### 3. Window

Slice every approved video's regions into overlapping 5s training windows and rebuild `data/manifest.csv`:

```bash
carzam window
```

### 4. Train

```bash
carzam train --config config/train_contrastive_v11.yaml
```

Runs land in `runs/<timestamp>/` with `checkpoint.pt`, configs and metrics. Compare runs with `scripts/compare_runs.py`.

### 5. Infer

```bash
# Local file
carzam infer clip.wav --checkpoint runs/<ts>/checkpoint.pt

# YouTube
carzam infer "https://youtu.be/..." --checkpoint runs/<ts>/checkpoint.pt

# Live mic, Shazam-style
carzam listen --checkpoint runs/<ts>/checkpoint.pt
```

---

## Auto-labeling

For scaling past hand-review. Two verifiers gate every window:

1. **Visual** — YOLOv8 detects cars in sampled frames, DINOv2 compares them against your curated reference photos. Drops videos where the target car isn't visually present, or that show the wrong car.
2. **Audio** — silero-VAD strips voiceover, LAION-CLAP zero-shots "engine vs. music/speech/noise", harmonicity rejects pure tones.

### One-time: build a reference library

```
data/references/
  ferrari_488/
    front-red.jpg
    side-yellow.jpg
    rear-track.jpg
    ...
  porsche_gt3rs/
    ...
  _interior/                # POV / cockpit / dashboard shots — shared across all cars
    cockpit-1.jpg
    ...
```

About **15 photos per car** is enough (mix angles & colors; background is fine — YOLO crops). The `_interior/` bucket lets us accept POV/cockpit footage with no exterior visible. Adding a class is just dropping a new folder — no retraining.

### Run it

```bash
# Single video
carzam auto-label "https://youtube.com/watch?v=..." --car ferrari_488

# Whole sources.yaml
carzam auto-label-batch --skip-existing

# Sanity-check a video locally without download cost
carzam visual-check path/to/video.mp4 --car ferrari_488
```

---

## Dataset

This repo **does not ship the audio dataset** — it's ~18GB of raw + windowed clips, and most of it is derivable from public YouTube videos. What ships is the *recipe*:

| What | Where | Size | Status |
|------|-------|------|--------|
| Source URL list | [`config/sources.yaml`](./config/sources.yaml) | small | in git |
| Hand-labeled regions (71 cars) | `data/regions/` | ~2.5MB | in git |
| Reference photos for auto-label | `data/references/` | ~6.5MB | in git |
| Manifest CSV (~40k windows) | `data/manifest.csv` | small | in git |
| Raw WAVs | `data/raw/` | ~5GB | **rebuild** |
| Training windows | `data/windows/` | ~6GB | **rebuild** |
| Caches | `data/*_cache/` | ~7GB | **rebuild** |
| Model checkpoints | `runs/`, `weights/` | varies | **rebuild** |

Rebuild from scratch:

```bash
carzam download && carzam window
```

YouTube videos get taken down occasionally. If a URL 404s, drop a comment in `sources.yaml` and find a replacement.

---

## Development

```bash
uv sync --extra dev --extra train --extra auto-label

uv run pytest
uv run ruff check .
uv run mypy src/
```

Python 3.11+. Tested on macOS (MPS) and Linux (CUDA). Training the contrastive model on the full corpus takes a few hours on a single GPU; an MPS Mac Studio can do it overnight.

### Contributing

Issues and PRs welcome. If you add a new car class, ideally include:
- A few source URLs in `config/sources.yaml`
- ~15 reference photos in `data/references/<your_car>/`
- The auto-label run's output in `data/regions/<your_car>/`

---

## License

[MIT](./LICENSE) © 2026 Jio Gallardy
