import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich.console import Console
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from carzam.data.dataset import CarAudioDataset
from carzam.data.manifest import read_manifest
from carzam.data.splits import split_by_video
from carzam.models.multihead import (
    CARS,
    ENGINE_FAMILIES,
    STATES,
    CarAudioModel,
)
from carzam.train import collate, pick_device

console = Console()


def _plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(6, len(labels))))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate(run_dir: Path, split: str = "test") -> None:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    rows = read_manifest(cfg["paths"]["manifest"])
    splits = split_by_video(
        rows,
        ratios=tuple(cfg["splits"]["ratios"]),
        seed=cfg["splits"]["seed"],
    )
    chosen = {"train": splits.train, "val": splits.val, "test": splits.test}[split]

    # Read classes.json to know what classes were used (and whether the model
    # was trained in families-as-classes mode).
    classes_path = run_dir / "classes.json"
    cars = list(CARS)
    use_families_as_classes = False
    if classes_path.exists():
        cdata = json.loads(classes_path.read_text())
        cars = list(cdata["cars"])
        use_families_as_classes = bool(cdata.get("families_as_classes", False))

    ds = CarAudioDataset(chosen, train=False, use_families_as_classes=use_families_as_classes)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)

    device = pick_device()
    sd = torch.load(run_dir / "checkpoint.pt", map_location=device)
    n_families = sd["family_head.weight"].shape[0] if "family_head.weight" in sd else None
    model = CarAudioModel(
        weights_path=None,
        n_cars=len(cars),
        n_families=n_families,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    car_true, car_pred, state_true, state_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logmel = batch["logmel"].to(device)
            cl, sl, _ = model(logmel)
            car_pred.extend(cl.argmax(1).cpu().tolist())
            state_pred.extend(sl.argmax(1).cpu().tolist())
            car_true.extend(batch["car_idx"].tolist())
            state_true.extend(batch["state_idx"].tolist())

    car_report = classification_report(
        car_true, car_pred,
        labels=list(range(len(cars))),
        target_names=list(cars),
        output_dict=True,
        zero_division=0,
    )
    state_report = classification_report(
        state_true, state_pred,
        labels=list(range(len(STATES))),
        target_names=list(STATES),
        output_dict=True,
        zero_division=0,
    )
    metrics = {"car": car_report, "state": state_report, "split": split}
    (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))

    cm_car = confusion_matrix(car_true, car_pred, labels=list(range(len(cars))))
    cm_state = confusion_matrix(state_true, state_pred, labels=list(range(len(STATES))))
    _plot_confusion(cm_car, list(cars), run_dir / f"confusion_car_{split}.png")
    _plot_confusion(cm_state, list(STATES), run_dir / f"confusion_state_{split}.png")

    console.print(
        f"[bold]{split}[/bold]  "
        f"car_acc={car_report['accuracy']:.3f}  "
        f"state_acc={state_report['accuracy']:.3f}"
    )
