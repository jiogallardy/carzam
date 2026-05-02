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

from carai.data.dataset import CarAudioDataset
from carai.data.manifest import read_manifest
from carai.data.splits import split_by_video
from carai.models.multihead import CarAudioModel

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
    }


def make_optimizer(model: CarAudioModel, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
            {"params": model.car_head.parameters(), "lr": cfg["train"]["lr_head"]},
            {"params": model.state_head.parameters(), "lr": cfg["train"]["lr_head"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )


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


def run_epoch(
    model: CarAudioModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)
    car_loss_sum = state_loss_sum = 0.0
    car_correct = state_correct = total = 0
    for batch in loader:
        logmel = batch["logmel"].to(device)
        car_y = batch["car_idx"].to(device)
        state_y = batch["state_idx"].to(device)
        with torch.set_grad_enabled(is_train):
            car_logits, state_logits = model(logmel)
            car_loss = F.cross_entropy(car_logits, car_y)
            state_loss = F.cross_entropy(state_logits, state_y)
            loss = (
                cfg["train"]["car_loss_weight"] * car_loss
                + cfg["train"]["state_loss_weight"] * state_loss
            )
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

    train_ds = CarAudioDataset(splits.train, train=True, seed=cfg["splits"]["seed"])
    val_ds = CarAudioDataset(splits.val, train=False)

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
    model = CarAudioModel(weights_path if weights_path.exists() else None).to(device)
    optimizer = make_optimizer(model, cfg)

    run_dir = Path(cfg["paths"]["runs_dir"]) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

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
        tr = run_epoch(model, train_loader, optimizer, device, cfg)
        va = (
            run_epoch(model, val_loader, None, device, cfg)
            if len(val_ds) > 0
            else EpochStats(0.0, 0.0, 0.0, 0.0)
        )
        elapsed = time.time() - ts
        console.print(
            f"epoch {epoch:02d}  "
            f"train: car_loss={tr.car_loss:.3f} car_acc={tr.car_acc:.3f} "
            f"state_acc={tr.state_acc:.3f}  "
            f"val: car_acc={va.car_acc:.3f} state_acc={va.state_acc:.3f}  "
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
