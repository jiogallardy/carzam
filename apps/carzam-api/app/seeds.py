"""Static metadata for the model's car classes. Re-run on every startup; idempotent.

The DB stores a UNION of every class id any model has ever output. The
recognize endpoint looks up the predicted id against this table to get a
display name. Adding new ids here is always safe; removing an id that an
older deployed model still emits would break that model's lookups.
"""

CAR_CLASS_SEEDS = [
    # ─────────────────────────────────────────────────────────────────────
    # v6 and earlier: specific car classes (8 fine-grained labels).
    # Kept here so the v6 checkpoint can still be selected from the picker.
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "ferrari_812",
        "display_name": "Ferrari 812 Superfast",
        "description": "6.5L naturally-aspirated V12 front-engine grand tourer.",
        "engine_family": "V12 NA",
        "sort_order": 10,
    },
    {
        "id": "lamborghini_huracan",
        "display_name": "Lamborghini Huracán",
        "description": "5.2L naturally-aspirated V10 mid-engine.",
        "engine_family": "V10 NA",
        "sort_order": 20,
    },
    {
        "id": "porsche_gt3",
        "display_name": "Porsche 911 GT3",
        "description": "4.0L NA flat-6. Includes 992 GT3 / GT3 RS / Cayman GT4 RS (shared engine family).",
        "engine_family": "Flat-6 NA",
        "sort_order": 30,
    },
    {
        "id": "amg_c63_m177",
        "display_name": "Mercedes-AMG C63 (M177)",
        "description": "4.0L twin-turbo V8.",
        "engine_family": "V8 TT",
        "sort_order": 40,
    },
    {
        "id": "bmw_m3_s58",
        "display_name": "BMW M3 / M4 (S58)",
        "description": "3.0L twin-turbo inline-6.",
        "engine_family": "I6 TT",
        "sort_order": 50,
    },
    {
        "id": "subaru_wrx_sti",
        "display_name": "Subaru WRX STI",
        "description": "Turbocharged flat-4 (EJ25).",
        "engine_family": "Flat-4 Turbo",
        "sort_order": 60,
    },
    {
        "id": "civic_type_r_k20c1",
        "display_name": "Honda Civic Type R (K20C1)",
        "description": "2.0L turbo inline-4.",
        "engine_family": "I4 Turbo",
        "sort_order": 70,
    },

    # ─────────────────────────────────────────────────────────────────────
    # v7: engine-family classes (12 broad labels). Trained on a wider clip
    # corpus and grouped at the family level for better generalization.
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "na_v12",
        "display_name": "Naturally-Aspirated V12",
        "description": "High-revving NA V12s — Ferrari 812, Aston DB12 V12, Lambo Aventador.",
        "engine_family": "V12 NA",
        "sort_order": 110,
    },
    {
        "id": "na_v10",
        "display_name": "Naturally-Aspirated V10",
        "description": "NA V10 mid-engine cars — Lamborghini Huracán, Audi R8 V10.",
        "engine_family": "V10 NA",
        "sort_order": 120,
    },
    {
        "id": "na_v8_flat",
        "display_name": "Naturally-Aspirated V8 (Flat-Plane)",
        "description": "Flat-plane crank NA V8s — Ferrari 458, McLaren 12C, S65/S63, Mustang GT350.",
        "engine_family": "V8 NA Flat",
        "sort_order": 130,
    },
    {
        "id": "na_flat6",
        "display_name": "Naturally-Aspirated Flat-6",
        "description": "Porsche GT-line NA flat-6 — 991/992 GT3, GT3 RS, Cayman GT4 RS.",
        "engine_family": "Flat-6 NA",
        "sort_order": 140,
    },
    {
        "id": "tt_v8_flat",
        "display_name": "Twin-Turbo V8 (Flat-Plane)",
        "description": "Flat-plane TT V8s — Ferrari F8, McLaren 720S, Maserati MC20.",
        "engine_family": "V8 TT Flat",
        "sort_order": 150,
    },
    {
        "id": "tt_v8_cross",
        "display_name": "Twin-Turbo V8 (Cross-Plane)",
        "description": "Cross-plane TT V8s — AMG M177/M178, BMW S63, Bentley/Audi 4.0 TT.",
        "engine_family": "V8 TT Cross",
        "sort_order": 160,
    },
    {
        "id": "supercharged_v8",
        "display_name": "Supercharged V8",
        "description": "Roots/screw blower V8s — Hellcat, Shelby GT500, Jaguar 5.0 SC.",
        "engine_family": "V8 SC",
        "sort_order": 170,
    },
    {
        "id": "tt_v6_hybrid",
        "display_name": "Twin-Turbo V6 Hybrid",
        "description": "Hybrid TT V6 supercars — Ferrari 296, McLaren Artura, NSX.",
        "engine_family": "V6 TT Hybrid",
        "sort_order": 180,
    },
    {
        "id": "tt_inline6",
        "display_name": "Twin-Turbo Inline-6",
        "description": "TT I6 — BMW S55/S58, Toyota Supra, AMG M256.",
        "engine_family": "I6 TT",
        "sort_order": 190,
    },
    {
        "id": "turbo_inline4",
        "display_name": "Turbocharged Inline-4",
        "description": "Single-turbo I4 — Civic Type R K20C1, EVO X 4B11T, Focus RS.",
        "engine_family": "I4 Turbo",
        "sort_order": 200,
    },
    {
        "id": "turbo_boxer4",
        "display_name": "Turbocharged Boxer-4",
        "description": "Subaru EJ20/EJ25 turbo flat-4 — WRX STI.",
        "engine_family": "Flat-4 Turbo",
        "sort_order": 210,
    },

    # Always last.
    {
        "id": "other",
        "display_name": "Other / Unknown",
        "description": "Fallback when confidence is low or the engine isn't one we trained on.",
        "engine_family": None,
        "sort_order": 999,
    },
]
