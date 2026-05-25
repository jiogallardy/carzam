"""Multi-model registry. Loads checkpoints from R2 on demand and keeps a
small LRU cache hot in memory so we don't OOM on a 4 GB Hetzner box.

Convention: each model lives under a folder in R2:
    s3://<bucket>/model/<id>/checkpoint.pt
    s3://<bucket>/model/<id>/classes.json

The legacy `s3://<bucket>/model/checkpoint.pt` (no folder) is exposed as
id `default` so we don't break existing deploys.
"""
from __future__ import annotations

import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from carzam.audio import load_wav, resample_to  # noqa: F401  (load_wav re-exported)
from carzam.data.dataset import SAMPLE_RATE, compute_logmel
from carzam.models.multihead import CarAudioModel

from app import storage
from app.config import settings


_DEVICE = torch.device("cpu")  # Hetzner box has no GPU; CPU is fine for CNN14

WINDOW_SECONDS = 5.0
HOP_SECONDS = 2.5             # window stride for sliding-window aggregate
OTHER_THRESHOLD = 0.40        # confidence floor for the single-window endpoint
SILENCE_RMS_THRESHOLD = 0.005

# Aggregate (Shazam-style) thresholds. Raw softmax probs across 72 classes
# top out around 0.20-0.30 even for confident predictions, so we use a
# combined absolute-floor + margin-over-runner-up rule. Three knobs:
#
#   * AGG_MIN_WINDOWS_FOR_CONFIDENT — single 5s shot is too noisy. Require
#     >=3 windows aggregated (=> >=10s of audio) before committing.
#   * AGG_CONFIDENT_MIN_TOP1 — absolute floor on top-1 probability.
#   * AGG_CONFIDENT_MARGIN_RATIO — top-1 must beat top-2 by this much.
#
# Random-noise calibration: random clips hit top-1 ~0.10-0.18 and margins
# ~1.3-1.7 by chance, so we need both gates to be stricter than that.
AGG_MIN_WINDOWS_FOR_CONFIDENT = 3
AGG_CONFIDENT_MIN_TOP1 = 0.18
AGG_CONFIDENT_MARGIN_RATIO = 1.7
AGG_UNKNOWN_MAX_TOP1 = 0.08          # below this top-1 -> "couldn't place"
AGG_UNKNOWN_MAX_MARGIN = 1.25        # if top1/top2 < this -> "couldn't place"

# Hot LRU cap. Each CNN14 checkpoint is ~318 MB; 2 models ≈ 700 MB resident.
# A CX22 (4 GB) handles 2 comfortably; bump if you upsize the box.
MODEL_CACHE_MAX = 2

LEGACY_DEFAULT_ID = "default"


@dataclass
class LoadedModel:
    id: str
    model: CarAudioModel
    cars: tuple[str, ...]
    states: tuple[str, ...]


# id -> LoadedModel. OrderedDict gives us O(1) LRU semantics via move_to_end.
_cache: "OrderedDict[str, LoadedModel]" = OrderedDict()
_default_id: str = LEGACY_DEFAULT_ID


def _r2_keys_for(model_id: str) -> tuple[str, str]:
    """Return (checkpoint_key, classes_key) for a given model id."""
    if model_id == LEGACY_DEFAULT_ID:
        cfg = settings()
        return cfg.model_r2_key, cfg.classes_r2_key
    return f"model/{model_id}/checkpoint.pt", f"model/{model_id}/classes.json"


def _local_paths_for(model_id: str) -> tuple[Path, Path]:
    cfg = settings()
    if model_id == LEGACY_DEFAULT_ID:
        return cfg.checkpoint_path, cfg.classes_path
    base = cfg.checkpoint_path.parent / model_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "checkpoint.pt", base / "classes.json"


def _ensure_local(model_id: str) -> tuple[Path, Path]:
    """Download checkpoint + classes from R2 to a local cache path.

    If the local files exist, we compare R2's ETag against the locally-stored
    ETag and re-download when they differ. Previously this was a pure
    "if not exists" gate, which caused stale checkpoints to live forever once
    cached on disk — re-uploading the same R2 key would be a no-op on the
    container until someone manually deleted /data/checkpoint.pt.
    """
    ckpt_key, classes_key = _r2_keys_for(model_id)
    ckpt_path, classes_path = _local_paths_for(model_id)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _sync(key: str, local: Path) -> None:
        etag_marker = local.with_suffix(local.suffix + ".etag")
        remote_etag: str | None = None
        try:
            remote_etag = storage.head_etag(key)
        except Exception:
            # If we can't reach R2 to check, fall back to the cached copy
            # rather than blowing up startup.
            remote_etag = None
        local_etag = etag_marker.read_text().strip() if etag_marker.exists() else None
        if local.exists() and remote_etag is not None and remote_etag == local_etag:
            return  # cached copy is current
        if local.exists() and remote_etag is None:
            return  # offline / R2 unreachable, keep what we have
        storage.download_to_path(key, str(local))
        if remote_etag:
            etag_marker.write_text(remote_etag)

    _sync(ckpt_key, ckpt_path)
    _sync(classes_key, classes_path)
    return ckpt_path, classes_path


def _load(model_id: str) -> LoadedModel:
    ckpt_path, classes_path = _ensure_local(model_id)
    classes = json.loads(Path(classes_path).read_text())
    cars = tuple(classes["cars"])
    states = tuple(classes["states"])

    # Auto-detect optional heads from the saved state dict so this loader
    # handles every model generation (v6 fine-only, v7 families-as-classes,
    # v9+ family+contrastive). Without this, strict load fails on any model
    # that has heads the constructor wasn't told about.
    state = torch.load(ckpt_path, map_location=_DEVICE, weights_only=True)
    n_families = (
        state["family_head.weight"].shape[0]
        if "family_head.weight" in state else None
    )
    # The contrastive embedding head is a small MLP — its first Linear lives
    # at embedding_head.0 in our nn.Sequential. classes.json also carries
    # embedding_dim for contrastive runs; prefer that, fall back to state.
    embedding_dim = classes.get("embedding_dim")
    if embedding_dim is None and "embedding_head.2.weight" in state:
        embedding_dim = state["embedding_head.2.weight"].shape[0]

    model = CarAudioModel(
        weights_path=None,
        n_cars=len(cars),
        n_states=len(states),
        n_families=n_families,
        embedding_dim=embedding_dim,
    ).to(_DEVICE)
    model.load_state_dict(state)
    model.eval()
    return LoadedModel(id=model_id, model=model, cars=cars, states=states)


def _get(model_id: str) -> LoadedModel:
    if model_id in _cache:
        _cache.move_to_end(model_id)
        return _cache[model_id]
    loaded = _load(model_id)
    _cache[model_id] = loaded
    while len(_cache) > MODEL_CACHE_MAX:
        evicted_id, _ = _cache.popitem(last=False)
        # No explicit free needed; refcount drops to 0 once the dict entry is gone.
        del evicted_id
    return loaded


def load_model() -> None:
    """Called once at startup. Pulls the default checkpoint and warms it."""
    global _default_id
    cfg = settings()
    # Honor an explicit override if the operator wants a versioned default.
    _default_id = cfg.default_model_id or LEGACY_DEFAULT_ID
    _get(_default_id)


def default_id() -> str:
    return _default_id


def model_run_id() -> str:
    """Back-compat shim — returns the configured run_id for /health."""
    return settings().model_run_id


def classes(model_id: str | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    loaded = _get(model_id or _default_id)
    return loaded.cars, loaded.states


def list_available() -> list[str]:
    """Discover model versions in R2. Always includes `default` first if the
    legacy path exists. Other versions are sorted by name (descending — so
    `v7` comes before `v6` and `20260601_*` before `20260507_*`).
    """
    ids: list[str] = []
    if storage.object_exists(settings().model_r2_key):
        ids.append(LEGACY_DEFAULT_ID)
    versioned = storage.list_subdirs("model/")
    # Filter to ones that actually have a checkpoint
    versioned = [v for v in versioned if storage.object_exists(f"model/{v}/checkpoint.pt")]
    versioned.sort(reverse=True)
    ids.extend(versioned)
    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def is_loaded(model_id: str) -> bool:
    return model_id in _cache


def _wav_bytes_to_window(wav_bytes: bytes) -> torch.Tensor:
    """Decode WAV bytes → 5s window tensor at 16kHz mono. Pads or center-crops as needed."""
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
    target_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(audio) < int(2.0 * SAMPLE_RATE):
        raise ValueError(f"clip too short: {len(audio) / SAMPLE_RATE:.1f}s, need >= 2s")
    if len(audio) < target_len:
        audio = np.concatenate([audio, np.zeros(target_len - len(audio), dtype=np.float32)])
    elif len(audio) > target_len:
        offset = (len(audio) - target_len) // 2
        audio = audio[offset:offset + target_len]
    return torch.from_numpy(np.ascontiguousarray(audio))


def _wav_bytes_to_mono_array(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV → float32 mono at SAMPLE_RATE (32 kHz). Resamples if needed.
    Returns the full audio without trimming/padding so the caller can slide
    its own windows."""
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE).astype(np.float32, copy=False)


def predict_bytes(wav_bytes: bytes, model_id: str | None = None) -> dict[str, Any]:
    """Run inference. If model_id is None, uses the configured default."""
    chosen_id = model_id or _default_id
    loaded = _get(chosen_id)

    audio = _wav_bytes_to_window(wav_bytes)

    rms = float(torch.sqrt(torch.mean(audio**2)).item())
    if rms < SILENCE_RMS_THRESHOLD:
        return {
            "predicted_class_id": None,
            "predicted_confidence": 0.0,
            "predicted_state": loaded.states[0] if loaded.states else "idle",
            "state_confidence": 0.0,
            "top3": [],
            "all_probs": {},
            "model_run_id": settings().model_run_id,
            "model_id": chosen_id,
            "silence": True,
        }

    logmel = compute_logmel(audio).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        # CarAudioModel.forward returns 3 outputs since the engine-family
        # head was added: (car_logits, state_logits, family_logits | None).
        # Older checkpoints (v6) trained without the family head still return
        # the 3-tuple — family_logits is None in that case.
        out = loaded.model(logmel)
        car_logits, state_logits = out[0], out[1]
        car_probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()
        state_probs = F.softmax(state_logits, dim=-1).squeeze(0).cpu().numpy()

    car_top1 = int(np.argmax(car_probs))
    state_top1 = int(np.argmax(state_probs))
    car_conf = float(car_probs[car_top1])

    above_threshold = car_conf >= OTHER_THRESHOLD
    matched_class = loaded.cars[car_top1] if above_threshold else None

    car_order = np.argsort(-car_probs)
    top3 = [(loaded.cars[i], float(car_probs[i])) for i in car_order[:3]]

    all_probs = {loaded.cars[i]: float(car_probs[i]) for i in range(len(loaded.cars))}

    return {
        "predicted_class_id": matched_class,
        "predicted_confidence": car_conf,
        "predicted_state": loaded.states[state_top1],
        "state_confidence": float(state_probs[state_top1]),
        "top3": top3,
        "all_probs": all_probs,
        "model_run_id": settings().model_run_id,
        "model_id": chosen_id,
    }


def predict_bytes_aggregate(
    wav_bytes: bytes, model_id: str | None = None,
) -> dict[str, Any]:
    """Sliding-window inference for Shazam-style continuous listening.

    Slides 5s windows at 2.5s hop across the input audio, runs the
    classifier on each, averages softmax probabilities, returns the
    aggregated top guess + a confidence verdict suitable for the mobile
    "is the answer stable yet" UX.

    For clips shorter than ~5s we fall back to single-window prediction.
    Empty / silent clips return is_silent=True.
    """
    chosen_id = model_id or _default_id
    loaded = _get(chosen_id)

    audio = _wav_bytes_to_mono_array(wav_bytes)
    if len(audio) < int(2.0 * SAMPLE_RATE):
        raise ValueError(f"clip too short: {len(audio) / SAMPLE_RATE:.1f}s, need >= 2s")

    # Whole-clip silence detection up front — saves window-by-window cycles.
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms < SILENCE_RMS_THRESHOLD:
        return {
            "predicted_class_id": None,
            "predicted_display_name": None,
            "predicted_confidence": 0.0,
            "is_confident": False,
            "couldnt_place": False,
            "n_windows_aggregated": 0,
            "duration_seconds": len(audio) / SAMPLE_RATE,
            "top3": [],
            "model_run_id": settings().model_run_id,
            "model_id": chosen_id,
            "silence": True,
        }

    win = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop = int(HOP_SECONDS * SAMPLE_RATE)

    # Build the list of window start offsets. For <5s clips we use a
    # single zero-padded window so we still return a meaningful answer.
    if len(audio) < win:
        starts = [0]
        padded = np.concatenate([audio, np.zeros(win - len(audio), dtype=np.float32)])
        windows = [padded]
    else:
        starts = list(range(0, len(audio) - win + 1, hop))
        if not starts:
            starts = [0]
        windows = [audio[s:s + win] for s in starts]

    # Run inference on each window and accumulate softmax probabilities.
    n_cars = len(loaded.cars)
    sum_probs = np.zeros(n_cars, dtype=np.float64)
    n_windows = 0
    for w in windows:
        w_rms = float(np.sqrt(np.mean(w * w)))
        if w_rms < SILENCE_RMS_THRESHOLD:
            continue  # skip silent windows so they don't dilute the average
        logmel = compute_logmel(torch.from_numpy(np.ascontiguousarray(w))).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            out = loaded.model(logmel)
            car_logits = out[0]
            probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()
        sum_probs += probs.astype(np.float64)
        n_windows += 1

    if n_windows == 0:
        # Entire clip was silence under the per-window RMS gate.
        return {
            "predicted_class_id": None,
            "predicted_display_name": None,
            "predicted_confidence": 0.0,
            "is_confident": False,
            "couldnt_place": False,
            "n_windows_aggregated": 0,
            "duration_seconds": len(audio) / SAMPLE_RATE,
            "top3": [],
            "model_run_id": settings().model_run_id,
            "model_id": chosen_id,
            "silence": True,
        }

    avg_probs = (sum_probs / n_windows).astype(np.float32)
    car_order = np.argsort(-avg_probs)
    car_top1 = int(car_order[0])
    top_conf = float(avg_probs[car_top1])
    top2_conf = float(avg_probs[car_order[1]]) if len(car_order) > 1 else 0.0
    margin = top_conf / max(top2_conf, 1e-6)

    top3 = [(loaded.cars[i], float(avg_probs[i])) for i in car_order[:3]]

    # Combined verdict — absolute floor + margin + min-windows requirement.
    is_confident = (
        n_windows >= AGG_MIN_WINDOWS_FOR_CONFIDENT
        and top_conf >= AGG_CONFIDENT_MIN_TOP1
        and margin >= AGG_CONFIDENT_MARGIN_RATIO
    )
    couldnt_place = (
        top_conf < AGG_UNKNOWN_MAX_TOP1
        or margin < AGG_UNKNOWN_MAX_MARGIN
    )
    predicted = loaded.cars[car_top1] if not couldnt_place else None

    return {
        "predicted_class_id": predicted,
        "predicted_confidence": top_conf,
        "is_confident": is_confident,
        "couldnt_place": couldnt_place,
        "n_windows_aggregated": n_windows,
        "duration_seconds": len(audio) / SAMPLE_RATE,
        "top3": top3,
        "model_run_id": settings().model_run_id,
        "model_id": chosen_id,
    }
