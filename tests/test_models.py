import torch

from carzam.models.backbone import build_backbone
from carzam.models.multihead import CARS, STATES, CarAudioModel


def test_backbone_output_shape():
    model = build_backbone(weights_path=None)
    model.eval()
    x = torch.randn(2, 501, 64)
    with torch.no_grad():
        emb = model(x)
    assert emb.shape == (2, 2048)


def test_class_constants():
    assert len(CARS) == 22
    assert "other" in CARS
    assert "ferrari_812" in CARS
    assert "ferrari_f12" in CARS
    assert "ferrari_458" in CARS
    assert "ferrari_488" in CARS
    assert "ferrari_f8" in CARS
    assert "ferrari_sf90" in CARS
    assert "ferrari_296" in CARS
    assert STATES == ("idle", "accel", "decel")


def test_multihead_forward_shapes():
    model = CarAudioModel(weights_path=None)
    model.eval()
    x = torch.randn(3, 501, 64)
    with torch.no_grad():
        car_logits, state_logits, family_logits = model(x)
    assert car_logits.shape == (3, 22)
    assert state_logits.shape == (3, 3)
    assert family_logits is None  # no family head when n_families is None


def test_multihead_with_family_head():
    model = CarAudioModel(weights_path=None, n_families=12)
    model.eval()
    x = torch.randn(3, 501, 64)
    with torch.no_grad():
        car_logits, state_logits, family_logits = model(x)
    assert car_logits.shape == (3, 22)
    assert state_logits.shape == (3, 3)
    assert family_logits is not None
    assert family_logits.shape == (3, 12)


def test_multihead_grad_flows():
    model = CarAudioModel(weights_path=None)
    x = torch.randn(2, 501, 64)
    car_logits, state_logits, _ = model(x)
    loss = car_logits.sum() + state_logits.sum()
    loss.backward()
    assert model.car_head.weight.grad is not None
    assert model.state_head.weight.grad is not None
    assert any(p.grad is not None for p in model.backbone.parameters())
