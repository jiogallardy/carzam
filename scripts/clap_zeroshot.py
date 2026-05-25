"""CLAP zero-shot vs trained PANNs cascade — engine-family classification.

Runs both algorithms on the *same* test split (split_by_video, seed=42) and
prints/saves a side-by-side classification report.

Why: experiment from the user's brainstorm — see whether a pretrained
audio-text foundation model (laion/clap-htsat-fused) is competitive with
the project's fine-tuned PANNs Cnn14 + family head, without any training.

Requires (not in pyproject.toml — experiment-only):
    uv pip install "transformers>=4.45"

Usage:
    .venv/bin/python scripts/clap_zeroshot.py \\
        --run-dir runs/20260507_171602 \\
        --out runs/clap_zeroshot

Notes:
- The trained model predicts fine-grained cars; we sum the softmax over
  cars that share an engine family to get a family probability. That is
  the natural "cascade stage 1" comparison the user wanted.
- CLAP wants 48kHz mono float audio. We resample on the fly.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from carzam.audio import load_wav, resample_to
from carzam.data.dataset import SAMPLE_RATE, compute_logmel
from carzam.data.manifest import read_manifest
from carzam.data.splits import split_by_video
from carzam.models.multihead import (
    CAR_TO_FAMILY,
    ENGINE_FAMILIES,
    CarAudioModel,
)

CLAP_SR = 48_000


# Prompt template per engine family. Mixes a generic phrasing
# ("the sound of a ... engine revving") with example car names — CLAP was
# trained on web audio-text pairs so brand names carry signal.
FAMILY_PROMPTS: dict[str, list[str]] = {
    "na_v12": [
        "the sound of a naturally aspirated V12 engine revving",
        "a Ferrari 812 V12 engine accelerating",
        "high-revving naturally aspirated twelve cylinder car engine",
    ],
    "na_v8_flat": [
        "a naturally aspirated flat-plane V8 engine revving",
        "Ferrari 458 V8 engine accelerating",
        "high-revving flat-plane crank V8 sports car engine",
    ],
    "tt_v8_flat": [
        "a twin-turbocharged flat-plane V8 engine revving",
        "Ferrari 488 or McLaren 720S twin-turbo V8 engine accelerating",
        "turbocharged flat-plane V8 supercar engine",
    ],
    "tt_v6_hybrid": [
        "a twin-turbocharged V6 hybrid engine revving",
        "Ferrari 296 GTB twin-turbo V6 hybrid engine accelerating",
        "small displacement turbocharged V6 hybrid sports car engine",
    ],
    "na_v10": [
        "a naturally aspirated V10 engine revving",
        "Lamborghini Huracan or Audi R8 V10 engine accelerating",
        "high-revving ten cylinder supercar engine",
    ],
    "na_flat6": [
        "a naturally aspirated flat-six boxer engine revving",
        "Porsche 911 GT3 flat-six engine accelerating",
        "high-revving horizontally opposed six cylinder sports car engine",
    ],
    "tt_v8_cross": [
        "a twin-turbocharged cross-plane V8 engine revving",
        "AMG 4-litre twin-turbo V8 engine accelerating",
        "deep rumbling turbocharged cross-plane V8 engine",
    ],
    "tt_inline6": [
        "a twin-turbocharged inline-six engine revving",
        "BMW M3 S58 inline-six engine accelerating",
        "turbocharged straight-six performance car engine",
    ],
    "turbo_boxer4": [
        "a turbocharged flat-four boxer engine revving",
        "Subaru WRX STI EJ257 boxer engine accelerating",
        "rumbly turbocharged horizontally opposed four cylinder engine",
    ],
    "turbo_inline4": [
        "a turbocharged inline-four engine revving",
        "Honda Civic Type R K20C turbo four cylinder engine accelerating",
        "high-revving turbocharged four cylinder hot hatch engine",
    ],
    "supercharged_v8": [
        "a supercharged V8 engine revving",
        "Corvette ZR1 supercharged V8 engine accelerating",
        "whining supercharged eight cylinder American muscle engine",
    ],
    "other": [
        "a car engine",
        "an ordinary passenger car engine sound",
    ],
}


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_v7(
    run_dir: Path, device: torch.device
) -> tuple[CarAudioModel, list[str], bool]:
    cdata = json.loads((run_dir / "classes.json").read_text())
    cars = list(cdata["cars"])
    use_families_as_classes = bool(cdata.get("families_as_classes", False))
    sd = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    n_families = sd["family_head.weight"].shape[0] if "family_head.weight" in sd else None
    model = CarAudioModel(weights_path=None, n_cars=len(cars), n_families=n_families).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cars, use_families_as_classes


def _car_probs_to_family_probs(
    car_probs: np.ndarray,
    cars: list[str],
) -> np.ndarray:
    """Sum car softmax → family-level probability. (B, n_cars) → (B, n_families)."""
    fam_probs = np.zeros((car_probs.shape[0], len(ENGINE_FAMILIES)), dtype=np.float32)
    for ci, car in enumerate(cars):
        fi = ENGINE_FAMILIES.index(CAR_TO_FAMILY.get(car, "other"))
        fam_probs[:, fi] += car_probs[:, ci]
    return fam_probs


def _predict_v7_family(
    run_dir: Path,
    rows: list,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true_family_idx, y_pred_family_idx)."""
    model, cars, families_mode = _load_v7(run_dir, device)
    print(f"  loaded {run_dir.name} — {len(cars)} car classes, "
          f"family_head={'yes' if model.family_head is not None else 'no'}, "
          f"families_as_classes={families_mode}")

    y_true: list[int] = []
    y_pred: list[int] = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            audio, sr = load_wav(r.path)
            if sr != SAMPLE_RATE:
                audio = resample_to(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
            logmel = compute_logmel(torch.from_numpy(np.ascontiguousarray(audio)))
            logmel = logmel.unsqueeze(0).to(device)  # (1, frames, n_mels)
            car_logits, _, fam_logits = model(logmel)

            if families_mode:
                # head outputs family logits directly
                pred_idx = int(car_logits.argmax(1).item())
                # remap: the run's "cars" list IS engine families in this mode
                fam_name = cars[pred_idx]
                pred_fam = ENGINE_FAMILIES.index(fam_name) if fam_name in ENGINE_FAMILIES else \
                    ENGINE_FAMILIES.index("other")
            elif fam_logits is not None:
                pred_fam = int(fam_logits.argmax(1).item())
            else:
                car_probs = F.softmax(car_logits, dim=1).cpu().numpy()
                fam_probs = _car_probs_to_family_probs(car_probs, cars)
                pred_fam = int(np.argmax(fam_probs, axis=1)[0])

            true_fam = ENGINE_FAMILIES.index(CAR_TO_FAMILY.get(r.car, "other"))
            y_true.append(true_fam)
            y_pred.append(pred_fam)
            if (i + 1) % 50 == 0:
                print(f"    v7 {i+1}/{len(rows)}  ({time.time()-t0:.1f}s)")
    return np.array(y_true), np.array(y_pred)


def _build_clap_text_embeds(
    clap_model,
    clap_processor,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Returns (text_embeds [n_families x dim], ordered family names)."""
    family_names: list[str] = []
    prompts: list[str] = []
    prompt_counts: list[int] = []
    for fam in ENGINE_FAMILIES:
        ps = FAMILY_PROMPTS[fam]
        family_names.append(fam)
        prompts.extend(ps)
        prompt_counts.append(len(ps))

    inputs = clap_processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feat = clap_model.get_text_features(**inputs)
    # transformers >=5 wraps the projection in BaseModelOutputWithPooling
    if not torch.is_tensor(text_feat):
        text_feat = text_feat.pooler_output
    text_feat = F.normalize(text_feat, dim=-1).cpu().numpy()

    # Average prompts per family, then renormalize
    avg = np.zeros((len(family_names), text_feat.shape[1]), dtype=np.float32)
    cursor = 0
    for i, k in enumerate(prompt_counts):
        avg[i] = text_feat[cursor:cursor + k].mean(axis=0)
        cursor += k
    avg = avg / (np.linalg.norm(avg, axis=1, keepdims=True) + 1e-12)
    return avg, family_names


def _predict_clap_family(
    rows: list,
    clap_id: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    from transformers import ClapModel, ClapProcessor

    print(f"  loading CLAP: {clap_id}")
    clap_processor = ClapProcessor.from_pretrained(clap_id)
    clap_model = ClapModel.from_pretrained(clap_id).to(device)
    clap_model.eval()

    text_embeds, family_names = _build_clap_text_embeds(clap_model, clap_processor, device)
    print(f"  built text embeds: {text_embeds.shape}  families={family_names}")

    # MPS has occasional kernel gaps for ClapAudioModel — fall back to CPU
    # for the audio tower if we hit one.
    y_true: list[int] = []
    y_pred: list[int] = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            audio, sr = load_wav(r.path)
            if sr != CLAP_SR:
                audio = resample_to(audio, src_sr=sr, dst_sr=CLAP_SR)
            inputs = clap_processor(audio=[audio], sampling_rate=CLAP_SR, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            audio_feat = clap_model.get_audio_features(**inputs)
            if not torch.is_tensor(audio_feat):
                audio_feat = audio_feat.pooler_output
            audio_feat = F.normalize(audio_feat, dim=-1).cpu().numpy()[0]
            sims = text_embeds @ audio_feat  # (n_families,)
            pred_fam_name = family_names[int(np.argmax(sims))]
            pred_fam = ENGINE_FAMILIES.index(pred_fam_name)
            true_fam = ENGINE_FAMILIES.index(CAR_TO_FAMILY.get(r.car, "other"))
            y_true.append(true_fam)
            y_pred.append(pred_fam)
            if (i + 1) % 50 == 0:
                print(f"    clap {i+1}/{len(rows)}  ({time.time()-t0:.1f}s)")
    return np.array(y_true), np.array(y_pred)


def _report(y_true: np.ndarray, y_pred: np.ndarray, present_only: bool) -> dict:
    """Classification report. If present_only, restrict labels to those that
    appear in y_true (so absent classes don't dilute macro F1).
    Always inject overall accuracy — sklearn drops the key when labels= is set."""
    present = sorted(set(y_true.tolist())) if present_only else list(range(len(ENGINE_FAMILIES)))
    names = [ENGINE_FAMILIES[i] for i in present]
    rep = classification_report(
        y_true, y_pred, labels=present, target_names=names,
        output_dict=True, zero_division=0,
    )
    rep["accuracy"] = float(np.mean(y_true == y_pred))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="A run dir with checkpoint.pt + classes.json (the model baseline).")
    ap.add_argument("--clap-id", default="laion/clap-htsat-fused")
    ap.add_argument("--device", default=None,
                    help="torch device override (default auto: mps/cuda/cpu).")
    ap.add_argument("--out", default="runs/clap_zeroshot", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run on first N test rows (smoke testing).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    print(f"device: {device}")

    rows_all = read_manifest(args.manifest)
    sp = split_by_video(rows_all, ratios=tuple(args.ratios), seed=args.seed)
    test = sp.test
    if args.limit:
        test = test[:args.limit]
    print(f"test windows: {len(test)} (from manifest {args.manifest})")

    fam_counts = defaultdict(int)
    for r in test:
        fam_counts[CAR_TO_FAMILY.get(r.car, "other")] += 1
    print("test family supports:")
    for f, n in sorted(fam_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {f:20s} {n}")

    print("\n=== v7 (PANNs cascade stage 1) ===")
    v7_true, v7_pred = _predict_v7_family(args.run_dir, test, device)
    v7_report = _report(v7_true, v7_pred, present_only=True)

    print("\n=== CLAP zero-shot ===")
    try:
        clap_true, clap_pred = _predict_clap_family(test, args.clap_id, device)
    except (NotImplementedError, RuntimeError) as e:
        print(f"  device {device} failed for CLAP ({e}); retrying on cpu")
        clap_true, clap_pred = _predict_clap_family(test, args.clap_id, torch.device("cpu"))
    clap_report = _report(clap_true, clap_pred, present_only=True)

    # Save raw arrays + reports.
    out_payload = {
        "manifest": args.manifest,
        "run_dir": str(args.run_dir),
        "clap_id": args.clap_id,
        "seed": args.seed,
        "ratios": list(args.ratios),
        "n_test": len(test),
        "v7": {
            "accuracy": v7_report["accuracy"],
            "macro_f1": v7_report["macro avg"]["f1-score"],
            "weighted_f1": v7_report["weighted avg"]["f1-score"],
            "report": v7_report,
        },
        "clap": {
            "accuracy": clap_report["accuracy"],
            "macro_f1": clap_report["macro avg"]["f1-score"],
            "weighted_f1": clap_report["weighted avg"]["f1-score"],
            "report": clap_report,
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(out_payload, indent=2))
    np.savez(
        args.out / "preds.npz",
        v7_true=v7_true, v7_pred=v7_pred,
        clap_true=clap_true, clap_pred=clap_pred,
    )

    # Side-by-side stdout report.
    present = sorted(set(v7_true.tolist()) | set(clap_true.tolist()))
    print("\n" + "=" * 72)
    print(f"{'family':<22} {'support':>8} {'v7 F1':>8} {'CLAP F1':>10}")
    print("-" * 72)
    for fi in present:
        name = ENGINE_FAMILIES[fi]
        sup = int(np.sum(v7_true == fi))
        v7_f1 = v7_report.get(name, {}).get("f1-score", 0.0)
        clap_f1 = clap_report.get(name, {}).get("f1-score", 0.0)
        print(f"{name:<22} {sup:>8} {v7_f1:>8.3f} {clap_f1:>10.3f}")
    print("-" * 72)
    print(f"{'accuracy':<22} {len(v7_true):>8} {v7_report['accuracy']:>8.3f} "
          f"{clap_report['accuracy']:>10.3f}")
    print(f"{'macro F1':<22} {'':>8} {v7_report['macro avg']['f1-score']:>8.3f} "
          f"{clap_report['macro avg']['f1-score']:>10.3f}")
    print(f"{'weighted F1':<22} {'':>8} {v7_report['weighted avg']['f1-score']:>8.3f} "
          f"{clap_report['weighted avg']['f1-score']:>10.3f}")
    print("=" * 72)
    print(f"\nsaved → {args.out / 'metrics.json'}")

    # Confusion matrices (optional, only if matplotlib available).
    try:
        import matplotlib.pyplot as plt
        for tag, yt, yp in (("v7", v7_true, v7_pred), ("clap", clap_true, clap_pred)):
            cm = confusion_matrix(yt, yp, labels=present)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(present)))
            ax.set_yticks(range(len(present)))
            ax.set_xticklabels([ENGINE_FAMILIES[i] for i in present], rotation=45, ha="right")
            ax.set_yticklabels([ENGINE_FAMILIES[i] for i in present])
            for i in range(len(present)):
                for j in range(len(present)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{tag} engine-family test confusion")
            fig.tight_layout()
            fig.savefig(args.out / f"confusion_{tag}.png", dpi=120)
            plt.close(fig)
        print(f"confusion matrices → {args.out}/")
    except Exception as e:
        print(f"(skipped confusion plots: {e})")


if __name__ == "__main__":
    main()
