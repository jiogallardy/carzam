import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from rich.console import Console
from torch.utils.data import DataLoader

from collections import Counter

from carai.data.dataset import CarAudioDataset
from carai.data.manifest import read_manifest
from carai.data.splits import split_by_video
from carai.models.multihead import (
    CARS,
    ENGINE_FAMILIES,
    STATES,
    CarAudioModel,
    family_index_for_car,
)

console = Console()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    console.print("[yellow]no GPU found, using CPU[/yellow]")
    return torch.device("cpu")


def collate(batch: list[dict]) -> dict:
    mels = [b["logmel"] for b in batch]
    max_t = max(m.shape[0] for m in mels)
    n_mel = mels[0].shape[1]
    out = torch.zeros(len(mels), max_t, n_mel, dtype=torch.float32)
    for i, m in enumerate(mels):
        out[i, : m.shape[0]] = m
    return {
        "logmel": out,
        "car_idx": torch.tensor([b["car_idx"] for b in batch], dtype=torch.long),
        "state_idx": torch.tensor([b["state_idx"] for b in batch], dtype=torch.long),
        "family_idx": torch.tensor(
            [b.get("family_idx", 0) for b in batch], dtype=torch.long
        ),
    }


def make_optimizer(model: CarAudioModel, cfg: dict) -> torch.optim.Optimizer:
    groups = [
        {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
        {"params": model.car_head.parameters(), "lr": cfg["train"]["lr_head"]},
        {"params": model.state_head.parameters(), "lr": cfg["train"]["lr_head"]},
    ]
    if model.family_head is not None:
        groups.append(
            {"params": model.family_head.parameters(), "lr": cfg["train"]["lr_head"]}
        )
    return torch.optim.AdamW(groups, weight_decay=cfg["train"]["weight_decay"])


def cosine_lr(epoch: int, total: int, warmup: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class EpochStats:
    car_loss: float
    state_loss: float
    car_acc: float
    state_acc: float
    family_loss: float = 0.0
    family_acc: float = 0.0


def compute_class_weights(rows, classes: tuple[str, ...], attr: str) -> torch.Tensor:
    """Inverse-frequency weights, normalized so mean weight = 1."""
    counts = Counter(getattr(r, attr) for r in rows if r.label is not None)
    weights = []
    for c in classes:
        n = counts.get(c, 0)
        weights.append(1.0 / max(n, 1))
    w = torch.tensor(weights, dtype=torch.float32)
    return w * (len(classes) / w.sum())  # mean = 1


def run_epoch(
    model: CarAudioModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict,
    car_weights: torch.Tensor | None = None,
    family_weights: torch.Tensor | None = None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)
    car_loss_sum = state_loss_sum = family_loss_sum = 0.0
    car_correct = state_correct = family_correct = total = 0
    for batch in loader:
        logmel = batch["logmel"].to(device)
        car_y = batch["car_idx"].to(device)
        state_y = batch["state_idx"].to(device)
        family_y = batch["family_idx"].to(device)
        with torch.set_grad_enabled(is_train):
            car_logits, state_logits, family_logits = model(logmel)
            car_loss = F.cross_entropy(car_logits, car_y, weight=car_weights)
            state_loss = F.cross_entropy(state_logits, state_y)
            loss = (
                cfg["train"]["car_loss_weight"] * car_loss
                + cfg["train"]["state_loss_weight"] * state_loss
            )
            if family_logits is not None:
                family_loss = F.cross_entropy(family_logits, family_y, weight=family_weights)
                fam_w = cfg["train"].get("family_loss_weight", 0.5)
                loss = loss + fam_w * family_loss
                family_loss_sum += family_loss.item() * logmel.size(0)
                family_correct += (family_logits.argmax(1) == family_y).sum().item()
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        car_loss_sum += car_loss.item() * logmel.size(0)
        state_loss_sum += state_loss.item() * logmel.size(0)
        car_correct += (car_logits.argmax(1) == car_y).sum().item()
        state_correct += (state_logits.argmax(1) == state_y).sum().item()
        total += logmel.size(0)
    return EpochStats(
        car_loss=car_loss_sum / total,
        state_loss=state_loss_sum / total,
        car_acc=car_correct / total,
        state_acc=state_correct / total,
        family_loss=family_loss_sum / total if total else 0.0,
        family_acc=family_correct / total if total else 0.0,
    )


def train(config_path: Path) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    torch.set_default_dtype(torch.float32)

    rows = read_manifest(cfg["paths"]["manifest"])
    splits = split_by_video(
        rows,
        ratios=tuple(cfg["splits"]["ratios"]),
        seed=cfg["splits"]["seed"],
    )
    console.print(
        f"split: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}"
    )

    use_families_as_classes = cfg["train"].get("use_families_as_classes", False)
    if use_families_as_classes:
        console.print("[bold yellow]families-as-classes mode: training on 12 engine families[/bold yellow]")
    train_ds = CarAudioDataset(
        splits.train, train=True, seed=cfg["splits"]["seed"],
        use_families_as_classes=use_families_as_classes,
    )
    val_ds = CarAudioDataset(
        splits.val, train=False,
        use_families_as_classes=use_families_as_classes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
    )

    device = pick_device()
    weights_path = Path(cfg["paths"]["weights"])
    # Family head only makes sense when fine classes != family classes.
    use_family = (
        cfg["train"].get("family_loss_weight", 0.0) > 0.0
        and not use_families_as_classes
    )
    active_classes = ENGINE_FAMILIES if use_families_as_classes else CARS
    model = CarAudioModel(
        weights_path if weights_path.exists() else None,
        n_cars=len(active_classes),
        n_families=len(ENGINE_FAMILIES) if use_family else None,
    ).to(device)
    optimizer = make_optimizer(model, cfg)

    # Class weights for the (active) car head.
    if use_families_as_classes:
        from collections import Counter
        fam_counts = Counter(family_index_for_car(r.car) for r in splits.train if r.label is not None)
        cw = torch.tensor(
            [1.0 / max(fam_counts.get(i, 0), 1) for i in range(len(ENGINE_FAMILIES))],
            dtype=torch.float32,
        )
        car_weights = (cw * (len(ENGINE_FAMILIES) / cw.sum())).to(device)
    else:
        car_weights = compute_class_weights(splits.train, CARS, "car").to(device)
    console.print(
        f"active classes ({len(active_classes)}): "
        + ", ".join(f"{c}={w:.2f}" for c, w in zip(active_classes, car_weights.cpu().tolist()))
    )

    family_weights = None
    if use_family:
        # Inverse-frequency weighting at the family level
        from collections import Counter
        fam_counts = Counter(family_index_for_car(r.car) for r in splits.train if r.label is not None)
        fam_w = torch.tensor(
            [1.0 / max(fam_counts.get(i, 0), 1) for i in range(len(ENGINE_FAMILIES))],
            dtype=torch.float32,
        )
        family_weights = (fam_w * (len(ENGINE_FAMILIES) / fam_w.sum())).to(device)
        console.print(f"using hierarchical family head ({len(ENGINE_FAMILIES)} families)")

    run_dir = Path(cfg["paths"]["runs_dir"]) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    classes_payload = {"cars": list(active_classes), "states": list(STATES)}
    if use_family:
        classes_payload["families"] = list(ENGINE_FAMILIES)
    if use_families_as_classes:
        classes_payload["families_as_classes"] = True
    (run_dir / "classes.json").write_text(json.dumps(classes_payload, indent=2))

    base_lrs = [
        cfg["train"]["lr_backbone"],
        cfg["train"]["lr_head"],
        cfg["train"]["lr_head"],
    ]
    best_val = -1.0
    patience = 0
    history: list[dict] = []
    for epoch in range(cfg["train"]["max_epochs"]):
        scale = cosine_lr(epoch, cfg["train"]["max_epochs"], cfg["train"]["warmup_epochs"])
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

        ts = time.time()
        tr = run_epoch(
            model, train_loader, optimizer, device, cfg,
            car_weights=car_weights, family_weights=family_weights,
        )
        va = (
            run_epoch(
                model, val_loader, None, device, cfg,
                car_weights=car_weights, family_weights=family_weights,
            )
            if len(val_ds) > 0
            else EpochStats(0.0, 0.0, 0.0, 0.0)
        )
        elapsed = time.time() - ts
        fam_str = f" fam_acc={va.family_acc:.3f}" if use_family else ""
        console.print(
            f"epoch {epoch:02d}  "
            f"train: car_loss={tr.car_loss:.3f} car_acc={tr.car_acc:.3f}  "
            f"val: car_acc={va.car_acc:.3f}{fam_str}  "
            f"({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch, "train": tr.__dict__, "val": va.__dict__})
        if va.car_acc > best_val:
            best_val = va.car_acc
            patience = 0
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
            console.print(f"  [green]saved checkpoint (val car_acc={best_val:.3f})[/green]")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                console.print(f"[yellow]early stop at epoch {epoch}[/yellow]")
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    console.print(f"[bold]done.[/bold] run dir: {run_dir}")
    return run_dir
