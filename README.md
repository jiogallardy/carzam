# carAI

Audio classifier that identifies car make + engine state (idle/accel/decel) from a short WAV clip.

## Setup

```bash
uv sync
bash scripts/fetch_panns_weights.sh
```

## Run

```bash
carai download
carai window
carai label
carai train
carai infer path/to/clip.wav
```

See `docs/superpowers/specs/2026-05-01-carai-design.md` for design and `docs/superpowers/plans/2026-05-02-carai-implementation.md` for the implementation plan.
