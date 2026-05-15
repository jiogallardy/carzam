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
    "mclaren_720s",      # M840T 4.0L TT V8 (720S / 750S — base body)
    "mclaren_765lt",     # M840T 4.0L TT V8 (765LT — track-focused longtail variant)
    "mclaren_senna",     # M840T 4.0L TT V8 (Senna — most aggressive variant + GTR)
    "audi_r8_v10",       # 5.2L NA V10 — same physical engine as Lambo Huracán
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

# Coarse engine families (for hierarchical training).
# Many fine classes share an engine architecture; the coarse head provides
# stable supervision while the fine head learns the harder model-level split.
ENGINE_FAMILIES = (
    "na_v12",          # Ferrari 812 / F12
    "na_v8_flat",      # Ferrari 458 (NA V8, flat-plane)
    "tt_v8_flat",      # Ferrari 488/F8/SF90 + McLaren 720s/765lt/Senna (TT V8, flat-plane)
    "tt_v6_hybrid",    # Ferrari 296 (TT V6, hybrid)
    "na_v10",          # Audi R8 V10 + Lambo Huracán (literally same engine)
    "na_flat6",        # Porsche GT3 / GT4 (mid- and rear-engine NA flat-6)
    "tt_v8_cross",     # AMG C63 M177 (TT V8, cross-plane crank)
    "tt_inline6",      # BMW M3/M4 S58
    "turbo_boxer4",    # Subaru WRX STI EJ257
    "turbo_inline4",   # Civic Type R K20C1
    "supercharged_v8", # Corvette ZR1/Z06 (C6/C7) — supercharged 6.2L V8
    "other",
)

# Map fine class name -> engine family. Keep in sync with CARS above.
CAR_TO_FAMILY: dict[str, str] = {
    "ferrari_812":         "na_v12",
    "ferrari_f12":         "na_v12",
    "ferrari_458":         "na_v8_flat",
    "ferrari_488":         "tt_v8_flat",
    "ferrari_f8":          "tt_v8_flat",
    "ferrari_sf90":        "tt_v8_flat",     # also has hybrid but engine architecture is TT V8 flat-plane
    "mclaren_720s":        "tt_v8_flat",
    "mclaren_765lt":       "tt_v8_flat",
    "mclaren_senna":       "tt_v8_flat",
    "audi_r8_v10":         "na_v10",
    "lamborghini_huracan": "na_v10",
    "ferrari_296":         "tt_v6_hybrid",
    "porsche_gt3":         "na_flat6",
    "porsche_gt4":         "na_flat6",
    "amg_c63_m177":        "tt_v8_cross",
    "bmw_m3_s58":          "tt_inline6",
    "subaru_wrx_sti":      "turbo_boxer4",
    "civic_type_r_k20c1":  "turbo_inline4",
    "corvette_c6_zr1":     "supercharged_v8",
    "corvette_c7_zr1":     "supercharged_v8",
    "corvette_c7_z06":     "supercharged_v8",
    "other":               "other",
}


def family_index_for_car(car: str) -> int:
    family = CAR_TO_FAMILY.get(car, "other")
    return ENGINE_FAMILIES.index(family)


BACKBONE_DIM = 2048


class CarAudioModel(nn.Module):
    def __init__(
        self,
        weights_path: Path | str | None,
        n_cars: int = len(CARS),
        n_states: int = len(STATES),
        n_families: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(weights_path)
        self.car_head = nn.Linear(BACKBONE_DIM, n_cars)
        self.state_head = nn.Linear(BACKBONE_DIM, n_states)
        # Optional coarse engine-family head (hierarchical training).
        # When loaded from older checkpoints without this head, n_families is None.
        self.family_head: nn.Linear | None = (
            nn.Linear(BACKBONE_DIM, n_families) if n_families is not None else None
        )
        # Optional projection head for contrastive / metric learning.
        # Produces L2-normalized 128-D embeddings used for nearest-prototype
        # inference and open-set rejection. Pre-existing checkpoints don't
        # have this — load_state_dict with strict=False keeps them working.
        self.embedding_dim = embedding_dim
        self.embedding_head: nn.Sequential | None = None
        if embedding_dim is not None:
            self.embedding_head = nn.Sequential(
                nn.Linear(BACKBONE_DIM, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, embedding_dim),
            )

        nn.init.xavier_uniform_(self.car_head.weight)
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.zeros_(self.car_head.bias)
        nn.init.zeros_(self.state_head.bias)
        if self.family_head is not None:
            nn.init.xavier_uniform_(self.family_head.weight)
            nn.init.zeros_(self.family_head.bias)
        if self.embedding_head is not None:
            for m in self.embedding_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self, logmel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (car_logits, state_logits, family_logits).

        Kept stable for backwards compatibility with infer/eval. Callers that
        want the embedding should use `forward_with_embedding` instead.
        """
        emb = self.backbone(logmel)
        car_logits = self.car_head(emb)
        state_logits = self.state_head(emb)
        family_logits = self.family_head(emb) if self.family_head is not None else None
        return car_logits, state_logits, family_logits

    def forward_with_embedding(
        self, logmel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Same as forward but also returns the L2-normalized embedding when
        an embedding head is configured. Returns None for the embedding when
        the model is in classifier-only mode (so callers can detect that case)."""
        emb = self.backbone(logmel)
        car_logits = self.car_head(emb)
        state_logits = self.state_head(emb)
        family_logits = self.family_head(emb) if self.family_head is not None else None
        z = None
        if self.embedding_head is not None:
            z = self.embedding_head(emb)
            z = torch.nn.functional.normalize(z, dim=-1)
        return car_logits, state_logits, family_logits, z

    @torch.no_grad()
    def embed(self, logmel: torch.Tensor) -> torch.Tensor:
        """Convenience: return the L2-normalized embedding only. Errors if
        the model wasn't built with an embedding head."""
        if self.embedding_head is None:
            raise RuntimeError(
                "model has no embedding head — load a contrastive-trained checkpoint"
            )
        emb = self.backbone(logmel)
        z = self.embedding_head(emb)
        return torch.nn.functional.normalize(z, dim=-1)
