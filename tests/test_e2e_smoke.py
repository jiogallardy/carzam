import numpy as np
import soundfile as sf
import torch
import yaml

from carzam.data.manifest import ManifestRow, write_manifest
from carzam.infer import predict_clip
from carzam.models.multihead import CARS
from carzam.train import train as train_run


def make_clip(path, sr: int = 16000, freq: float = 200.0) -> None:
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, x, sr)


def test_train_then_infer(tmp_path):
    win_dir = tmp_path / "windows"
    win_dir.mkdir()
    rows = []
    for ci, car in enumerate(CARS[:2]):
        for v in range(2):
            for i in range(4):
                p = win_dir / f"{car}_v{v}_{i}.wav"
                make_clip(p, freq=200.0 + 50 * ci + 5 * i)
                rows.append(
                    ManifestRow(
                        path=str(p),
                        car=car,
                        source_video=f"{car}_v{v}",
                        start_time=float(i) * 2.5,
                        label="idle" if i % 2 == 0 else "accel",
                    )
                )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    cfg = {
        "paths": {
            "manifest": str(manifest),
            "weights": "weights/does_not_exist.pth",
            "runs_dir": str(tmp_path / "runs"),
        },
        "splits": {"ratios": [0.6, 0.2, 0.2], "seed": 0},
        "train": {
            "batch_size": 4,
            "num_workers": 0,
            "max_epochs": 2,
            "early_stop_patience": 5,
            "lr_head": 1e-3,
            "lr_backbone": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "car_loss_weight": 1.0,
            "state_loss_weight": 1.0,
        },
    }
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = train_run(cfg_path)
    assert (run_dir / "checkpoint.pt").exists()

    clip = tmp_path / "mystery.wav"
    make_clip(clip)
    result = predict_clip(clip, run_dir / "checkpoint.pt", device=torch.device("cpu"))
    assert result["car"] in list(CARS) + ["other"]
    assert result["state"] in ("idle", "accel", "decel")
