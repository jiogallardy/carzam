"""Wraps the carai inference module so our endpoints don't reach into Torch directly.
The model loads ONCE at app startup and stays resident in process memory.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from carai.audio import load_wav, resample_to
from carai.data.dataset import SAMPLE_RATE, compute_logmel
from carai.models.multihead import CarAudioModel

from app import storage
from app.config import settings


_DEVICE = torch.device("cpu")  # Hetzner box has no GPU; CPU is fine for CNN14
_model: CarAudioModel | None = None
_cars: tuple[str, ...] = ()
_states: tuple[str, ...] = ()
_model_run_id: str = "unknown"

WINDOW_SECONDS = 5.0
OTHER_THRESHOLD = 0.40


def _ensure_local_files() -> None:
    """Pull checkpoint + classes from R2 if they aren't on disk yet.
    The /data dir is a persistent volume in production; first boot downloads,
    subsequent boots skip.
    """
    cfg = settings()
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.checkpoint_path.exists():
        storage.download_to_path(cfg.model_r2_key, str(cfg.checkpoint_path))
    if not cfg.classes_path.exists():
        storage.download_to_path(cfg.classes_r2_key, str(cfg.classes_path))


def load_model() -> None:
    """Called once at startup. Pulls checkpoint from R2 (first boot), then loads."""
    global _model, _cars, _states, _model_run_id
    cfg = settings()
    _ensure_local_files()
    classes = json.loads(Path(cfg.classes_path).read_text())
    _cars = tuple(classes["cars"])
    _states = tuple(classes["states"])
    model = CarAudioModel(weights_path=None, n_cars=len(_cars), n_states=len(_states)).to(_DEVICE)
    state = torch.load(cfg.checkpoint_path, map_location=_DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _model = model
    _model_run_id = cfg.model_run_id


def classes() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _cars, _states


def model_run_id() -> str:
    return _model_run_id


def _wav_bytes_to_window(wav_bytes: bytes) -> torch.Tensor:
    """Decode WAV bytes → 5s window tensor at 16kHz mono. Pads or center-crops as needed."""
    # carai.audio.load_wav expects a path; we use soundfile directly here.
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
    target_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(audio) < int(2.0 * SAMPLE_RATE):
        raise ValueError(f"clip too short: {len(audio) / SAMPLE_RATE:.1f}s, need >= 2s")
    if len(audio) < target_len:
        audio = np.concatenate([audio, np.zeros(target_len - len(audio), dtype=np.float32)])
    elif len(audio) > target_len:
        # center crop — same as production, avoids edge-of-clip artifacts
        offset = (len(audio) - target_len) // 2
        audio = audio[offset:offset + target_len]
    return torch.from_numpy(np.ascontiguousarray(audio))


def predict_bytes(wav_bytes: bytes) -> dict[str, Any]:
    """Run the model on raw WAV bytes. Returns predictions dict with all_probs included."""
    if _model is None:
        raise RuntimeError("model not loaded; call load_model() at startup")
    audio = _wav_bytes_to_window(wav_bytes)
    logmel = compute_logmel(audio).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        car_logits, state_logits = _model(logmel)
        car_probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()
        state_probs = F.softmax(state_logits, dim=-1).squeeze(0).cpu().numpy()

    car_top1 = int(np.argmax(car_probs))
    state_top1 = int(np.argmax(state_probs))
    car_conf = float(car_probs[car_top1])

    above_threshold = car_conf >= OTHER_THRESHOLD
    matched_class = _cars[car_top1] if above_threshold else None

    car_order = np.argsort(-car_probs)
    top3 = [(_cars[i], float(car_probs[i])) for i in car_order[:3]]

    all_probs = {_cars[i]: float(car_probs[i]) for i in range(len(_cars))}

    return {
        "predicted_class_id": matched_class,
        "predicted_confidence": car_conf,
        "predicted_state": _states[state_top1],
        "state_confidence": float(state_probs[state_top1]),
        "top3": top3,
        "all_probs": all_probs,
        "model_run_id": _model_run_id,
    }
