from pathlib import Path

import torch

from carzam.models.cnn14 import Cnn14


def build_backbone(weights_path: Path | str | None) -> Cnn14:
    model = Cnn14()
    if weights_path is None:
        return model
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    own = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing = set(own.keys()) - set(loaded.keys())
    own.update(loaded)
    model.load_state_dict(own)
    if missing:
        print(f"[backbone] missing {len(missing)} keys (heads, expected)")
    return model
