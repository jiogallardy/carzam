"""v8 cascade inference.

Stage 1: family classifier (v7, families-as-classes) picks an engine family.
Stage 2: family-specific specialist picks the specific car within that family.

For single-car families (e.g. amg_c63_m177 is the only car in tt_v8_cross),
stage 2 is skipped and the family prediction itself is the final answer
mapped back to the single car name.

Specialist checkpoints live under runs/v8/<run-ts>/ — discovered by reading
their classes.json for the `specialist_family` key. The cascade loader builds
a {family_name: LoadedSpecialist} map at init.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from carzam.data.dataset import SAMPLE_RATE, compute_logmel
from carzam.models.multihead import CAR_TO_FAMILY, CARS, ENGINE_FAMILIES, CarAudioModel


@dataclass
class LoadedSpecialist:
    family: str
    cars: tuple[str, ...]
    model: CarAudioModel


def _load_checkpoint(run_dir: Path, device: torch.device) -> tuple[CarAudioModel, dict]:
    classes = json.loads((run_dir / "classes.json").read_text())
    sd = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True)
    n_families = sd["family_head.weight"].shape[0] if "family_head.weight" in sd else None
    n_cars = len(classes["cars"])
    n_states = len(classes["states"])
    model = CarAudioModel(
        weights_path=None, n_cars=n_cars, n_states=n_states, n_families=n_families,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, classes


def discover_specialists(root: Path, device: torch.device) -> dict[str, LoadedSpecialist]:
    """Walk runs/v8 looking for subdirs whose classes.json declares a
    specialist_family. Keep the latest run-dir per family (sorted by name)."""
    by_family: dict[str, LoadedSpecialist] = {}
    if not root.exists():
        return by_family
    candidates: dict[str, Path] = {}
    for sub in sorted(root.iterdir()):  # sorted → later runs override earlier
        if not sub.is_dir():
            continue
        classes_path = sub / "classes.json"
        ckpt_path = sub / "checkpoint.pt"
        if not classes_path.exists() or not ckpt_path.exists():
            continue
        try:
            classes = json.loads(classes_path.read_text())
        except Exception:
            continue
        fam = classes.get("specialist_family")
        if not fam:
            continue
        candidates[fam] = sub
    for fam, sub in candidates.items():
        model, classes = _load_checkpoint(sub, device)
        by_family[fam] = LoadedSpecialist(
            family=fam, cars=tuple(classes["cars"]), model=model,
        )
    return by_family


def _family_to_single_car() -> dict[str, str]:
    """Map families that contain exactly one car to that car name. Used as
    a fast-path when no specialist is needed (e.g. tt_v8_cross → amg_c63_m177)."""
    by_fam: dict[str, list[str]] = {}
    for car, fam in CAR_TO_FAMILY.items():
        by_fam.setdefault(fam, []).append(car)
    return {fam: cars[0] for fam, cars in by_fam.items() if len(cars) == 1}


def predict_cascade(
    audio_window: torch.Tensor,
    family_model: CarAudioModel,
    family_classes: tuple[str, ...],
    specialists: dict[str, LoadedSpecialist],
    device: torch.device,
    other_threshold: float = 0.40,
) -> dict[str, Any]:
    """Run stage 1 (family) + stage 2 (within-family specialist) on a single
    5s audio window. Returns the final fine-class prediction plus debugging info.
    """
    single_car = _family_to_single_car()

    logmel = compute_logmel(audio_window).unsqueeze(0).to(device)
    with torch.no_grad():
        car_logits, _state_logits, _ = family_model(logmel)
        family_probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()

    top1_idx = int(np.argmax(family_probs))
    top1_family = family_classes[top1_idx]
    top1_family_conf = float(family_probs[top1_idx])

    # If the family classifier is too uncertain, fall back to "other".
    if top1_family_conf < other_threshold and top1_family != "other":
        return {
            "predicted_car": "other",
            "predicted_family": top1_family,
            "family_confidence": top1_family_conf,
            "car_confidence": top1_family_conf,
            "stage2_used": False,
            "top3_family": [
                (family_classes[i], float(family_probs[i]))
                for i in np.argsort(-family_probs)[:3]
            ],
            "top3_car": [],
        }

    # Single-car family → no stage 2.
    if top1_family in single_car:
        car = single_car[top1_family]
        return {
            "predicted_car": car,
            "predicted_family": top1_family,
            "family_confidence": top1_family_conf,
            "car_confidence": top1_family_conf,
            "stage2_used": False,
            "top3_family": [
                (family_classes[i], float(family_probs[i]))
                for i in np.argsort(-family_probs)[:3]
            ],
            "top3_car": [(car, top1_family_conf)],
        }

    # Multi-car family → consult the specialist.
    spec = specialists.get(top1_family)
    if spec is None:
        # No specialist trained yet for this family. Best we can do is name
        # the family; surface it via the family field.
        return {
            "predicted_car": None,
            "predicted_family": top1_family,
            "family_confidence": top1_family_conf,
            "car_confidence": top1_family_conf,
            "stage2_used": False,
            "top3_family": [
                (family_classes[i], float(family_probs[i]))
                for i in np.argsort(-family_probs)[:3]
            ],
            "top3_car": [],
        }

    with torch.no_grad():
        car_logits, _, _ = spec.model(logmel)
        car_probs = F.softmax(car_logits, dim=-1).squeeze(0).cpu().numpy()

    car_top1_idx = int(np.argmax(car_probs))
    car_top1 = spec.cars[car_top1_idx]
    car_conf = float(car_probs[car_top1_idx])
    # Joint confidence = family * within-family. Reasonable proxy for
    # how confident the chain is end-to-end.
    joint = top1_family_conf * car_conf

    car_order = np.argsort(-car_probs)
    top3_car = [(spec.cars[i], float(car_probs[i])) for i in car_order[:3]]
    return {
        "predicted_car": car_top1,
        "predicted_family": top1_family,
        "family_confidence": top1_family_conf,
        "car_confidence": car_conf,
        "joint_confidence": joint,
        "stage2_used": True,
        "top3_family": [
            (family_classes[i], float(family_probs[i]))
            for i in np.argsort(-family_probs)[:3]
        ],
        "top3_car": top3_car,
    }
