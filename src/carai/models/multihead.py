from pathlib import Path

import torch
import torch.nn as nn

from carai.models.backbone import build_backbone

# Class labels. porsche_gt3 (rear-engine 911 GT3/GT3 RS, 4.0L NA flat-6) is
# kept separate from porsche_gt4 (mid-engine Cayman GT4/GT4 RS) — even though
# the 982 GT4 RS uses literally the same engine as the 992 GT3, exhaust routing
# + cabin acoustics differ enough that the model picks up the distinction.
CARS = (
    "ferrari_812",       # NA V12 6.5L (Superfast/GTS/Competizione)
    "ferrari_f12",       # NA V12 6.3L (F12 Berlinetta/tdf, GTC4Lusso)
    "ferrari_458",       # NA V8 4.5L flat-plane (Italia/Speciale/Spider)
    "ferrari_488",       # TT V8 3.9L flat-plane (488 GTB/Pista/Spider)
    "ferrari_f8",        # TT V8 3.9L (F8 Tributo/Spider — same engine as 488)
    "ferrari_sf90",      # TT V8 + hybrid (SF90 Stradale)
    "ferrari_296",       # TT V6 + hybrid (296 GTB/GTS)
    "lamborghini_huracan",
    "porsche_gt3",
    "porsche_gt4",
    "amg_c63_m177",
    "bmw_m3_s58",
    "subaru_wrx_sti",
    "civic_type_r_k20c1",
    "corvette_c6_zr1",
    "corvette_c7_zr1",
    "corvette_c7_z06",
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
