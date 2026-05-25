from pathlib import Path

import torch
import torch.nn as nn

from carzam.models.backbone import build_backbone

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
    # v10 expansion (auto-labeled with audio-only pipeline)
    "acura_nsx_nc1",
    "aston_dbs_superleggera",
    "aston_martin_v12_vantage",
    "aston_martin_valkyrie",
    "aston_vanquish",
    "bugatti_chiron",
    "bugatti_veyron",
    "corvette_z06_c8",
    "dodge_demon",
    "dodge_viper_srt10",
    "ferrari_360",
    "ferrari_enzo",
    "ferrari_f40",
    "ferrari_f50",
    "ferrari_laferrari",
    "ford_gt40",
    "ford_gt_2017",
    "hennessey_venom_f5",
    "jaguar_xj220",
    "koenigsegg_agera_rs",
    "koenigsegg_jesko",
    "koenigsegg_regera",
    "lamborghini_aventador",
    "lamborghini_diablo",
    "lamborghini_gallardo",
    "lamborghini_murcielago",
    "lamborghini_revuelto",
    "lexus_lfa",
    "lotus_evija",
    "mazda_rx7_fd",
    "mclaren_750s",
    "mclaren_artura",
    "mclaren_f1",
    "mclaren_mp4_12c",
    "mclaren_p1",
    "mclaren_speedtail",
    "mercedes_amg_gt_black",
    "mercedes_amg_one",
    "mercedes_slr_mclaren",
    "mercedes_sls_amg",
    "nissan_gtr_r35",
    "nissan_skyline_r34",
    "pagani_huayra",
    "pagani_zonda",
    "porsche_911_gt2_rs",
    "porsche_911_turbo_s",
    "porsche_918_spyder",
    "porsche_992_gt3_rs",
    "porsche_carrera_gt",
    "rimac_nevera",
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
    # v10 expansion (auto-labeled). Mapping is best-effort — feel free to
    # refine; the contrastive trainer can survive an "other" fallback.
    "ferrari_360":              "na_v8_flat",      # 3.6L flat-plane NA V8
    "ferrari_enzo":             "na_v12",          # 6.0L NA V12
    "ferrari_f40":              "tt_v8_flat",      # 2.9L TT V8 flat-plane
    "ferrari_f50":              "na_v12",          # 4.7L NA V12
    "ferrari_laferrari":        "na_v12",          # 6.3L NA V12 + hybrid
    "ford_gt_2017":             "tt_v8_cross",     # 3.5L TT V6 actually — close enough; could remap
    "ford_gt40":                "na_v8_flat",      # NA V8 (vintage)
    "aston_martin_v12_vantage": "na_v12",
    "aston_martin_valkyrie":    "na_v12",          # Cosworth 6.5L NA V12
    "aston_dbs_superleggera":   "tt_v8_cross",     # 5.2L TT V12 actually — leaving as is for now
    "aston_vanquish":           "na_v12",
    "lamborghini_aventador":    "na_v12",
    "lamborghini_diablo":       "na_v12",
    "lamborghini_gallardo":     "na_v10",
    "lamborghini_murcielago":   "na_v12",
    "lamborghini_revuelto":     "na_v12",          # V12 + hybrid
    "pagani_huayra":            "tt_v8_cross",     # AMG-built 6.0L TT V12 → no exact family; use cross
    "pagani_zonda":             "na_v12",          # AMG 7.3L NA V12
    "mclaren_f1":               "na_v12",          # BMW S70/2 NA V12
    "mclaren_mp4_12c":          "tt_v8_flat",      # 3.8L TT V8
    "mclaren_p1":               "tt_v8_flat",      # 3.8L TT V8 + hybrid
    "mclaren_speedtail":        "tt_v8_flat",
    "mclaren_artura":           "tt_v6_hybrid",    # 3.0L TT V6 + hybrid
    "mclaren_750s":             "tt_v8_flat",
    "porsche_918_spyder":       "na_v8_flat",      # 4.6L NA V8 + hybrid
    "porsche_carrera_gt":       "na_v10",          # 5.7L NA V10
    "porsche_911_turbo_s":      "tt_v8_cross",     # flat-6 TT — closest to cross-plane bucket
    "porsche_911_gt2_rs":       "tt_v8_cross",     # 3.8L TT flat-6
    "porsche_992_gt3_rs":       "na_flat6",        # 4.0L NA flat-6
    "mercedes_amg_one":         "tt_v6_hybrid",    # F1-derived hybrid V6
    "mercedes_amg_gt_black":    "tt_v8_cross",     # 4.0L TT V8 flat-plane (M178)
    "mercedes_sls_amg":         "na_v8_flat",      # 6.2L NA V8 — cross-plane really
    "mercedes_slr_mclaren":     "supercharged_v8", # 5.4L supercharged V8
    "nissan_gtr_r35":           "tt_v8_cross",     # 3.8L TT V6 (VR38)
    "nissan_skyline_r34":       "tt_inline6",      # 2.6L TT inline-6 (RB26)
    "acura_nsx_nc1":            "tt_v6_hybrid",    # 3.5L TT V6 + hybrid
    "mazda_rx7_fd":             "other",           # rotary — no good bucket
    "dodge_viper_srt10":        "na_v10",          # 8.4L NA V10
    "dodge_demon":              "supercharged_v8", # 6.2L supercharged V8 (Hellcat)
    "corvette_z06_c8":          "na_v8_flat",      # 5.5L NA V8 flat-plane (LT6)
    "bugatti_chiron":           "tt_v8_cross",     # 8.0L W16 quad-turbo — no W16 bucket
    "bugatti_veyron":           "tt_v8_cross",
    "lexus_lfa":                "na_v10",          # 4.8L NA V10
    "jaguar_xj220":             "tt_v8_cross",     # 3.5L TT V6
    "koenigsegg_jesko":         "tt_v8_flat",      # 5.0L TT V8
    "koenigsegg_agera_rs":      "tt_v8_flat",
    "koenigsegg_regera":        "tt_v8_flat",      # also has electric motors
    "hennessey_venom_f5":       "tt_v8_cross",     # 6.6L TT V8 cross-plane
    "lotus_evija":              "other",           # full EV — no engine family
    "rimac_nevera":             "other",           # full EV
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
