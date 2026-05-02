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

See `docs/superpowers/specs/2026-05-01-carai-design.md` for design and `docs/superpowers/plans/2026-05-02-carai-implementation.md` for the implementation plan.
