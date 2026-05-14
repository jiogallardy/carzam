# carAI

Audio classifier that identifies car make + engine state (idle/accel/decel) from a short WAV clip.

## Setup

```bash
uv sync
bash scripts/fetch_panns_weights.sh
```

## Run

```bash
carai download                 # pulls all videos in config/sources.yaml as 16kHz mono WAV
carai review                   # interactive: play each video, mark labeled regions
carai window                   # auto-slice into 5s windows using regions
carai train                    # train multi-head model
carai infer path/to/clip.wav --checkpoint runs/<ts>/checkpoint.pt
```

### Reviewer keys

| key | action |
|-----|--------|
| `i` | start `idle` range |
| `a` | start `accel` range |
| `d` | start `decel` range |
| `s` | start `skip` range (intro / voiceover / music — gets dropped at windowing) |
| `x` | close current range, no new mode |
| `space` | pause / resume |
| `z` / `c` | rewind / forward 5s |
| `u` | undo last range |
| `q` | save & quit |
| `n` | next video without saving |

A new keypress closes any open range and starts the new mode.

### Optional fallback

`carai label` opens the older per-window labeler — useful only if you skipped review and want to label window-by-window.

## Auto-labeling pipeline

Scales labeling beyond hand-review by combining visual verification (YOLOv8 + DINOv2 against curated reference photos) with audio screening (silero VAD + LAION-CLAP zero-shot + harmonicity). Drops videos whose target car isn't visually present, then drops windows that contain voiceover / music / non-engine audio.

### One-time setup: reference photo library

```
data/references/
  ferrari_488/
    front-red.jpg
    side-yellow.jpg
    rear-track.jpg
    ...
  porsche_gt3rs/
    ...
  _interior/                # optional but recommended
    cockpit-1.jpg
    dashboard-1.jpg
    pov-driver-1.jpg
    ...
```

About **15 photos per car** is enough. Mix angles (front / side / rear / 3-quarter) and a few different colors. Photos can include background — we YOLO them to auto-crop the largest car. Adding a new class is just dropping a new folder; no retraining.

The `_interior/` bucket holds generic "you're inside a car" reference photos — steering wheels, dashboards, gauge clusters, POV/cockpit shots. When a frame has no exterior car visible, we check the whole frame against these references to distinguish **POV cockpit / dash cam / hood cam** (good audio source — accept) from **unrelated footage** (random pocket cam — reject). 10–20 generic shots covers all cars; the interior bucket isn't per-car.

### Single video

```bash
carai auto-label "https://www.youtube.com/watch?v=..." --car ferrari_488
```

Output: `data/regions/ferrari_488/<video_id>.yaml` with accepted ranges marked, ready for `carai window`.

### Batch over sources.yaml

```bash
carai auto-label-batch --skip-existing
```

Runs on every URL in `config/sources.yaml`. Builds the reference index once and reuses it for all videos.

### Sanity-check visuals without downloading audio

```bash
carai visual-check path/to/local-video.mp4 --car ferrari_488
```

Prints per-frame YOLO/DINOv2 verdicts and the final keep/drop decision. Use this to validate your reference library before kicking off a batch run.

See `docs/superpowers/specs/2026-05-01-carai-design.md` for design and `docs/superpowers/plans/2026-05-02-carai-implementation.md` for the implementation plan.
