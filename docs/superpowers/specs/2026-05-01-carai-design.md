# carAI — Design Document

**Date:** 2026-05-01
**Status:** Approved (pending spec review)

## Goal

Build an audio classifier that, given a 3–5 second clip of a car engine, predicts:

1. **Which car** it is (from a fixed list of 7 vehicles + an "other" class), and
2. **What state** the engine is in (idle / acceleration / deceleration).

v1 scope is intentionally narrow: pre-trimmed clips only, no segmentation of long recordings. Inference runs locally on Apple Silicon (M1 Max).

## Vehicle list (8 classes total)

| Class                | Engine config         | Why included                                  |
| -------------------- | --------------------- | --------------------------------------------- |
| `ferrari_812`        | NA V12                | Distinctive high-revving V12                  |
| `lamborghini_huracan`| NA V10                | V10 is acoustically unique                    |
| `porsche_gt3`        | NA flat-6             | Boxer-6 vs other 6-cyl is a real test         |
| `amg_c63_m177`       | TT V8                 | Modern turbo V8 character                     |
| `bmw_m3_s58`         | TT inline-6           | Forces model to discriminate turbo I6 vs V8   |
| `subaru_wrx_sti`     | Turbo flat-4 (boxer)  | Iconic boxer rumble; "spoiler" class          |
| `civic_type_r_k20c1` | Turbo inline-4        | Lower-bound / "normal car" baseline           |
| `other`              | —                     | Catch-all, prevents overconfident errors      |

The `other` class is seeded with ~100 clips of vehicles outside the list (Tesla, Camry, F-150, motorbike, diesel truck) so the model has a safe place to put unfamiliar input.

## Engine states (3 classes)

- `idle` — vehicle stationary, throttle off, RPM at idle
- `acceleration` — throttle on, RPM rising
- `deceleration` — throttle off with engine still running, RPM falling (overrun, where pops/crackles happen)

Cruise, cold start, and redline are excluded from v1 — they multiply labeling cost without proportional acoustic benefit.

## Inference contract

```
input:  WAV file or YouTube URL
output: { car: <class>, state: <class>, car_confidence: float, state_confidence: float }
```

- Clip < 2s → error.
- Clip 2–5s → silence-padded to 5s.
- Clip > 5s → center crop by default; `--all-windows` flag emits per-window timeline.
- If top-1 car confidence < 0.4, return `other` regardless of which class won.

## Architecture

Five CLI-driven stages, each consuming and producing files. No shared in-memory state — every stage is resumable.

```
sources.yaml ──► [download] ──► data/raw/<car>/*.wav
                                      │
                                      ▼
                              [window] ──► data/windows/<car>/*.wav
                                      │
                                      ▼
                              [labeler] ──► data/manifest.csv
                                      │
                                      ▼
                              [train]  ──► runs/<ts>/checkpoint.pt
                                      │
                                      ▼
                              [infer]  ──► JSON / pretty terminal output
```

### Module map

| Module                     | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| `carai.cli`                | Click-based dispatcher; subcommands `download`, `window`, `label`, `train`, `eval`, `infer` |
| `carai.data.download`      | `yt-dlp` wrapper. Pulls audio-only, transcodes to mono/16kHz WAV, idempotent |
| `carai.data.window`        | Sliding 5s windows, 2.5s hop, RMS silence filter           |
| `carai.data.labeler`       | Interactive CLI: plays window, captures keypress, writes manifest row |
| `carai.data.manifest`      | CSV read/write helpers for `manifest.csv`                  |
| `carai.data.splits`        | Stratified train/val/test split **by source video**, never by window |
| `carai.data.dataset`       | PyTorch `Dataset` + augmentations                          |
| `carai.models.backbone`    | Loads PANNs CNN14 weights, exposes 2048-dim embedding      |
| `carai.models.multihead`   | Backbone + `car_head` (Linear 2048→8) + `state_head` (Linear 2048→3) |
| `carai.train`              | Training loop, MPS-aware                                   |
| `carai.eval`               | Test-set evaluation, confusion matrices                    |
| `carai.infer`              | Single-clip and YouTube-URL inference                      |

## Model

**Backbone:** PANNs CNN14, pretrained on AudioSet (~80MB checkpoint).

**Why CNN14 over alternatives:**
- vs AST: ~5x faster on M1 Max, ~340MB smaller, ~90% as accurate for transfer learning on small datasets.
- vs from-scratch CNN: 1,500 clips is too few to learn engine acoustics from random init.
- vs YAMNet: TensorFlow dependency we don't want.

**Heads:**
- `car_head`: `Linear(2048 → 8)` — softmax over 7 cars + `other`
- `state_head`: `Linear(2048 → 3)` — softmax over idle/accel/decel

**Loss:** `total = car_loss + state_loss`, both cross-entropy, equal weight to start. Per-head loss tracked separately; re-weight if one plateaus.

**Input pipeline:**
- WAV resampled on-the-fly to 32kHz mono (PANNs native rate)
- 5-second window → log-mel spectrogram (64 mel bins, 501 frames)
- Storage at 16kHz to save disk; harmonics above 8kHz are not discriminative for this task

**Augmentations during training:**
- SpecAugment (freq + time masking)
- Random gain (±6dB)
- Random crop within the 5s window
- **Explicitly NOT:** pitch shift, time stretch — those corrupt engine fundamentals, which is the signal.

## Data pipeline

### Sources

`config/sources.yaml`:
```yaml
ferrari_812:
  - https://youtube.com/watch?v=...
  - ...
lamborghini_huracan:
  - ...
# ... per car
other:
  - ...
```

Sourcing strategy: **Claude populates this file by hand** during implementation, using web search + `yt-dlp ytsearch:` over known exotic-car channels (Marchettino, AutoTopNL, Shmee150, Mr.JWW, DailyDrivenExotics, Carfection). User reviews and prunes. No `discover` module in the codebase.

### Download

`carzam download` runs `yt-dlp` per URL: audio-only, transcoded to `mono / 16kHz / WAV`, written to `data/raw/<car>/<videoid>.wav`. Idempotent — skips files that already exist.

### Windowing

`carzam window` slices each raw file into 5-second windows with 2.5-second hop (50% overlap). Windows below an RMS silence threshold are discarded. Each surviving window becomes a row in a pre-label manifest with fields: `path, car, source_video, start_time, label=null`.

### Labeling

`carzam label` opens an interactive CLI loop:

```
[ferrari_812 / marchettino_2024.wav @ 47.5s ]  ▶ playing...
  (i)dle  (a)ccel  (d)ecel  (s)kip  (r)eplay  (q)uit
```

Keypress writes the row to `data/manifest.csv`. `s` (skip) marks the row as discarded. `q` exits cleanly; next session resumes where you left off.

**Targets:**
- ~50–80 windows per (car, state) cell
- ~1,200–1,700 labeled windows total
- ~3–5 hours of human labeling, broken into sessions

### Splits

`carai.data.splits` produces train/val/test (70/15/15) **by source video, not by window**. With ~10 videos per car this is roughly 7/2/1 per class. Stratification is by `car` only — `state` distribution is per-window and can't be enforced at video granularity, but with ~10 videos per car the state distribution will average out across splits in practice.

A unit test verifies that no `source_video` appears in more than one split — this is the bug that would silently inflate metrics, so we guard against it.

## Training

**Optimizer:** AdamW, weight decay 1e-4
**Learning rates:** `1e-4` for heads, `1e-5` for backbone (10x lower for pretrained weights)
**Schedule:** cosine annealing, 2-epoch warmup
**Batch size:** 32
**Early stopping:** val car-accuracy, patience 5 epochs
**Max epochs:** 50

### Apple Silicon (MPS) notes
- `torch.set_default_dtype(torch.float32)` — some PANNs ops drop to fp16 on MPS and produce NaNs.
- `num_workers=0` for DataLoader on MPS (multiprocessing + MPS hangs). Use a small in-memory cache.
- Verify `torch.backends.mps.is_available()` at startup; fall back to CPU with a warning.

### Run artifacts (`runs/<timestamp>/`)
- `metrics.json` — per-class precision/recall/F1 for both heads
- `confusion_car.png`, `confusion_state.png`
- `checkpoint.pt`
- `config.yaml` — frozen copy of hyperparameters

### Realistic expected outcome
- Car head top-1: **75–88%** on test set. Hardest confusions: AMG V8 ↔ S58 I6, Subaru flat-4 ↔ Civic I4.
- State head top-1: **85–92%**. Idle is easy; accel ↔ decel is the hard pair on steady mid-RPM clips.
- Training time on M1 Max: **20–40 minutes** for 30 epochs.

## Inference UX

```bash
$ carzam infer ~/Downloads/mystery_car.wav
─────────────────────────────────────────────
  carAI inference
  clip: mystery_car.wav  (4.8s, 16kHz mono)
─────────────────────────────────────────────
  car:    Ferrari 812         0.84
          Lamborghini Huracán 0.09
          Porsche GT3         0.04

  state:  acceleration        0.91
          deceleration        0.06
          idle                0.03
─────────────────────────────────────────────
  verdict: Ferrari 812, accelerating
```

```bash
$ carzam infer "https://youtube.com/watch?v=..."
# yt-dlp's the audio, takes the loudest 5s window, runs same pipeline

$ carzam infer clip.wav --json
{"car": "ferrari_812", "state": "acceleration", "car_confidence": 0.84, ...}

$ carzam infer clip.wav --all-windows
# emits per-window timeline for clips longer than 5s
```

## Repo layout

```
carAI/
├── pyproject.toml
├── README.md
├── .gitignore                  # data/, runs/, .venv/
├── config/
│   ├── sources.yaml
│   └── train.yaml
├── src/carzam/
│   ├── __init__.py
│   ├── cli.py
│   ├── data/
│   │   ├── download.py
│   │   ├── window.py
│   │   ├── labeler.py
│   │   ├── manifest.py
│   │   ├── splits.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── backbone.py
│   │   └── multihead.py
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── scripts/
│   └── fetch_panns_weights.sh
├── tests/
│   ├── test_window.py
│   ├── test_manifest.py
│   ├── test_splits.py
│   └── test_infer_smoke.py
└── data/                       # gitignored
    ├── raw/<car>/*.wav
    ├── windows/<car>/*.wav
    └── manifest.csv
```

## Dependencies & tooling

- **Package manager:** `uv` (fast, handles Python version + venv + deps).
- **Python:** `>=3.11`
- **Runtime deps:** `torch>=2.3`, `torchaudio>=2.3`, `numpy`, `pyyaml`, `click`, `rich`, `yt-dlp`, `soundfile`, `sounddevice`, `librosa`, `scikit-learn`, `matplotlib`, `tqdm`
- **Dev deps:** `pytest`, `ruff`, `mypy`
- **Entry point:** `carai = "carzam.cli:main"`

### First-run sequence

```bash
uv sync
bash scripts/fetch_panns_weights.sh
carzam download
carzam window
carzam label              # ~3–5 hours, resumable
carzam train
carzam infer clip.wav
```

## Out of scope for v1

- Long-clip segmentation (slide window across a 30s YouTube rip and emit a labeled timeline). Architecture supports it via `--all-windows`, but the user-facing "Shazam for cars" timeline UI is v2.
- Web UI / API server.
- Cloud training. Everything runs on the M1 Max.
- Auto-discovery of YouTube videos (`carai discover` was considered and dropped).
- Pitch-shift / time-stretch augmentation (would corrupt the signal we're learning).
- Cruise / cold-start / redline as separate state classes.

## Risks and mitigations

| Risk                                                     | Mitigation                                                  |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| YouTube audio quality varies (camera mics, voiceovers)   | Curated channel whitelist; manual labeler skips bad windows |
| Window-split leakage inflates metrics                    | Splits keyed by `source_video`; unit-tested                 |
| Model overconfidence on out-of-distribution input        | `other` class seeded with ~100 non-listed-vehicle clips; <0.4 confidence floor |
| MPS instability (NaNs, hangs)                            | fp32 forced; `num_workers=0`; CPU fallback                  |
| AMG V8 / BMW S58 confusion (similar turbo character)     | Acceptable for v1; tracked in confusion matrix              |
| Labeling fatigue (3–5 hours is a lot)                    | Resumable labeler; sessions of 30–45 min                    |
