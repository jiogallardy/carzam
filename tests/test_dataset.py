import numpy as np
import soundfile as sf
import torch

from carzam.data.dataset import CarAudioDataset, compute_logmel
from carzam.data.manifest import ManifestRow


def make_window(tmp_path, sr=16000) -> str:
    audio = (0.3 * np.random.RandomState(0).randn(sr * 5)).astype(np.float32)
    p = tmp_path / "x.wav"
    sf.write(p, audio, sr)
    return str(p)


def test_logmel_shape():
    audio = torch.randn(32000 * 5)
    mel = compute_logmel(audio, sample_rate=32000)
    assert mel.dim() == 2
    assert mel.shape[1] == 64
    assert mel.shape[0] > 100


def test_dataset_returns_correct_items(tmp_path):
    path = make_window(tmp_path)
    rows = [
        ManifestRow(path, "ferrari_812", "v1", 0.0, "accel"),
        ManifestRow(path, "porsche_gt3", "v2", 2.5, "idle"),
    ]
    ds = CarAudioDataset(rows, train=False)
    item = ds[0]
    assert "logmel" in item
    assert "car_idx" in item
    assert "state_idx" in item
    assert item["logmel"].shape[1] == 64
    assert item["car_idx"] == 0  # ferrari_812 is index 0
    assert item["state_idx"] == 1  # accel is index 1


def test_train_augmentations_change_output(tmp_path):
    path = make_window(tmp_path)
    rows = [ManifestRow(path, "ferrari_812", "v1", 0.0, "accel")]
    ds = CarAudioDataset(rows, train=True, seed=0)
    a = ds[0]["logmel"]
    ds = CarAudioDataset(rows, train=True, seed=1)
    b = ds[0]["logmel"]
    assert not torch.equal(a, b)
