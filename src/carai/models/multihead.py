from pathlib import Path

import torch
import torch.nn as nn

from carai.models.backbone import build_backbone

# Class labels. porsche_gt3 (rear-engine 911 GT3/GT3 RS, 4.0L NA flat-6) is
# kept separate from porsche_gt4 (mid-engine Cayman GT4/GT4 RS) — even though
# the 982 GT4 RS uses literally the same engine as the 992 GT3, exhaust routing
# + cabin acoustics differ enough that the model picks up the distinction.
CARS = (
    "ferrari_812",
    "lamborghini_huracan",
    "porsche_gt3",
    "porsche_gt4",
    "amg_c63_m177",
    "bmw_m3_s58",
    "subaru_wrx_sti",
    "civic_type_r_k20c1",
    "other",
)
STATES = ("idle", "accel", "decel")


class CarAudioModel(nn.Module):
    def __init__(
        self,
        weights_path: Path | str | None,
        n_cars: int = len(CARS),
        n_states: int = len(STATES),
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(weights_path)
        self.car_head = nn.Linear(2048, n_cars)
        self.state_head = nn.Linear(2048, n_states)
        nn.init.xavier_uniform_(self.car_head.weight)
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.zeros_(self.car_head.bias)
        nn.init.zeros_(self.state_head.bias)

    def forward(self, logmel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(logmel)
        return self.car_head(emb), self.state_head(emb)
