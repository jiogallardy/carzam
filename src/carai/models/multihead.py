from pathlib import Path

import torch
import torch.nn as nn

from carai.models.backbone import build_backbone

# Class labels.
# Note: porsche_gt3 lumps in Cayman GT4 / GT4 RS — they share the same NA flat-6
# engine family (GT4 RS literally uses the 992 GT3 engine). Tags in
# config/sources.yaml mark which physical model each URL is so the class can be
# split back into porsche_gt3 + porsche_gt4 if there's enough data.
CARS = (
    "ferrari_812",
    "lamborghini_huracan",
    "porsche_gt3",
    "amg_c63_m177",
    "bmw_m3_s58",
    "subaru_wrx_sti",
    "civic_type_r_k20c1",
    "other",
)
STATES = ("idle", "accel", "decel")


class CarAudioModel(nn.Module):
    def __init__(self, weights_path: Path | str | None) -> None:
        super().__init__()
        self.backbone = build_backbone(weights_path)
        self.car_head = nn.Linear(2048, len(CARS))
        self.state_head = nn.Linear(2048, len(STATES))
        nn.init.xavier_uniform_(self.car_head.weight)
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.zeros_(self.car_head.bias)
        nn.init.zeros_(self.state_head.bias)

    def forward(self, logmel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(logmel)
        return self.car_head(emb), self.state_head(emb)
