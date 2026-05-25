"""Embedding-side inference: prototypes, nearest-neighbor lookup, open-set.

The contrastive-trained model produces L2-normalized embeddings (128-D by
default). Inference uses a small table of "prototypes" — one vector per
known car, computed as the mean train-set embedding. At query time:
    sim(query, prototype_c) = dot(z_q, p_c)         # cosine since both unit-norm
    nearest = argmax_c sim
    if max_sim < threshold:  return "unknown"

Adding a new car at runtime = compute its prototype from a few clips and
append to the table. No retraining needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from carzam.audio import load_wav, resample_to
from carzam.data.dataset import SAMPLE_RATE, compute_logmel
from carzam.models.multihead import CARS, ENGINE_FAMILIES, STATES, CarAudioModel


# Default open-set threshold. Calibrated on a held-out class — see
# scripts/test_open_set_threshold.py once you have one. 0.55 is a
# reasonable starting point because SupCon-trained cosine sims for
# correct matches usually sit in [0.6, 0.95].
DEFAULT_UNKNOWN_THRESHOLD = 0.55


@dataclass
class PrototypeTable:
    cars: tuple[str, ...]
    embeddings: torch.Tensor           # (n_classes, D)
    n_samples: torch.Tensor            # (n_classes,) int — train rows averaged

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]


@dataclass
class MatchResult:
    nearest_car: str
    nearest_sim: float
    is_known: bool                     # max_sim >= threshold
    top_k: list[tuple[str, float]]


def load_prototypes(run_dir: Path) -> PrototypeTable:
    payload = torch.load(run_dir / "prototypes.pt", map_location="cpu", weights_only=False)
    return PrototypeTable(
        cars=tuple(payload["cars"]),
        embeddings=payload["embeddings"].float(),
        n_samples=payload["n_samples"],
    )


def load_embedding_model(run_dir: Path, device: torch.device) -> CarAudioModel:
    """Reload the contrastive checkpoint with the embedding head in place.

    Reads embedding_dim from classes.json so we instantiate the exact same
    architecture that was trained.
    """
    classes = json.loads((run_dir / "classes.json").read_text())
    embedding_dim = classes.get("embedding_dim")
    if embedding_dim is None:
        raise RuntimeError(
            f"{run_dir / 'classes.json'} has no embedding_dim — this isn't a "
            f"contrastive-trained run. Use carzam infer for classifier-only models."
        )
    n_cars = len(classes["cars"])
    sd = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True)
    n_families = sd["family_head.weight"].shape[0] if "family_head.weight" in sd else None
    model = CarAudioModel(
        weights_path=None, n_cars=n_cars, n_states=len(STATES),
        n_families=n_families, embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def embed_audio(
    audio: np.ndarray,
    sr: int,
    model: CarAudioModel,
    device: torch.device,
    window_seconds: float = 5.0,
    hop_seconds: float = 2.5,
) -> torch.Tensor:
    """Embed a clip. If it's longer than `window_seconds`, slide windows and
    average the embeddings (then re-normalize). Returns a (D,) tensor."""
    if sr != SAMPLE_RATE:
        audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
    win = int(window_seconds * SAMPLE_RATE)
    if len(audio) < win:
        # Pad short clips to one window
        pad = win - len(audio)
        audio = np.pad(audio, (0, pad))
    hop = int(hop_seconds * SAMPLE_RATE)
    starts = list(range(0, max(1, len(audio) - win + 1), hop)) or [0]
    embeddings: list[torch.Tensor] = []
    for s in starts:
        chunk = audio[s:s + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        logmel = compute_logmel(torch.from_numpy(chunk)).unsqueeze(0).to(device)
        z = model.embed(logmel)              # (1, D), unit-norm
        embeddings.append(z.squeeze(0).cpu())
    avg = torch.stack(embeddings).mean(dim=0)
    return torch.nn.functional.normalize(avg, dim=-1)


def match(
    z: torch.Tensor,
    prototypes: PrototypeTable,
    top_k: int = 5,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
) -> MatchResult:
    """Cosine-sim a query embedding against the prototype table."""
    if z.dim() != 1 or z.shape[0] != prototypes.dim:
        raise ValueError(f"shape mismatch: query {tuple(z.shape)} vs proto dim {prototypes.dim}")
    sims = prototypes.embeddings @ z         # (n_classes,)
    # Exclude classes with no train data (zero embeddings would give sim≈0)
    valid_mask = prototypes.n_samples > 0
    masked = sims.clone()
    masked[~valid_mask] = -1.0
    order = torch.argsort(masked, descending=True)
    top_idx = order[:top_k].tolist()
    top = [(prototypes.cars[i], float(masked[i])) for i in top_idx]
    best_idx = top_idx[0]
    best_sim = float(masked[best_idx])
    return MatchResult(
        nearest_car=prototypes.cars[best_idx],
        nearest_sim=best_sim,
        is_known=best_sim >= unknown_threshold,
        top_k=top,
    )


def predict_clip_embedding(
    clip_path: Path | str,
    run_dir: Path,
    device: torch.device,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
) -> dict:
    """End-to-end: load clip → embed → nearest prototype. Returns a dict
    suitable for JSON output. Mirrors the shape of infer.predict_clip but
    with embedding-specific fields.
    """
    model = load_embedding_model(run_dir, device)
    prototypes = load_prototypes(run_dir)

    audio, sr = load_wav(clip_path)
    z = embed_audio(audio, sr, model, device)
    result = match(z, prototypes, unknown_threshold=unknown_threshold)
    return {
        "clip": str(clip_path),
        "nearest_car": result.nearest_car,
        "nearest_sim": result.nearest_sim,
        "is_known": result.is_known,
        "verdict": result.nearest_car if result.is_known else "unknown",
        "top_k": result.top_k,
        "threshold": unknown_threshold,
        "embedding": z.tolist(),
    }
