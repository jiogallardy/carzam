"""Static metadata for the model's car classes. Re-run on every startup; idempotent."""

CAR_CLASS_SEEDS = [
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
    {
        "id": "other",
        "display_name": "Other / Unknown",
        "description": "Fallback when confidence is low or the engine isn't one we trained on.",
        "engine_family": None,
        "sort_order": 999,
    },
]
