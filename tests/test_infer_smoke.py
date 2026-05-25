import numpy as np
import soundfile as sf
import torch

from carzam.infer import predict_clip
from carzam.models.multihead import CarAudioModel


def test_predict_clip_returns_expected_shape(tmp_path):
    sr = 16000
    audio = (0.3 * np.random.RandomState(0).randn(sr * 5)).astype(np.float32)
    p = tmp_path / "clip.wav"
    sf.write(p, audio, sr)

    model = CarAudioModel(weights_path=None)
    model.eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt)

    result = predict_clip(p, ckpt, device=torch.device("cpu"))
    assert "car" in result and "state" in result
    assert "car_confidence" in result
    assert 0.0 <= result["car_confidence"] <= 1.0
    assert "car_top3" in result and len(result["car_top3"]) == 3
