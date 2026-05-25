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

    # ─────────────────────────────────────────────────────────────────────
    # v8–v10 expansion: extra Ferraris, McLarens, Porsches, AMC, BMW M3,
    # Corvettes — added during the cascade / contrastive experiments.
    # ─────────────────────────────────────────────────────────────────────
    {"id": "ferrari_f12", "display_name": "Ferrari F12 / GTC4Lusso", "description": "6.3L NA V12.", "engine_family": "V12 NA", "sort_order": 220},
    {"id": "ferrari_458", "display_name": "Ferrari 458", "description": "4.5L NA flat-plane V8.", "engine_family": "V8 NA Flat", "sort_order": 230},
    {"id": "ferrari_488", "display_name": "Ferrari 488", "description": "3.9L twin-turbo flat-plane V8.", "engine_family": "V8 TT Flat", "sort_order": 240},
    {"id": "ferrari_f8", "display_name": "Ferrari F8 Tributo", "description": "3.9L twin-turbo V8 (same engine as 488).", "engine_family": "V8 TT Flat", "sort_order": 250},
    {"id": "ferrari_sf90", "display_name": "Ferrari SF90 Stradale", "description": "TT V8 + hybrid plug-in.", "engine_family": "V8 TT Flat", "sort_order": 260},
    {"id": "ferrari_296", "display_name": "Ferrari 296 GTB", "description": "3.0L twin-turbo V6 hybrid.", "engine_family": "V6 TT Hybrid", "sort_order": 270},
    {"id": "mclaren_720s", "display_name": "McLaren 720S", "description": "4.0L twin-turbo V8 (M840T).", "engine_family": "V8 TT Flat", "sort_order": 280},
    {"id": "mclaren_765lt", "display_name": "McLaren 765LT", "description": "M840T longtail variant.", "engine_family": "V8 TT Flat", "sort_order": 285},
    {"id": "mclaren_senna", "display_name": "McLaren Senna", "description": "M840T-aggressive track variant.", "engine_family": "V8 TT Flat", "sort_order": 290},
    {"id": "audi_r8_v10", "display_name": "Audi R8 V10", "description": "5.2L NA V10 — same physical engine as Lambo Huracán.", "engine_family": "V10 NA", "sort_order": 300},
    {"id": "porsche_gt4", "display_name": "Porsche Cayman GT4 / GT4 RS", "description": "Mid-engine NA flat-6.", "engine_family": "Flat-6 NA", "sort_order": 310},
    {"id": "corvette_c6_zr1", "display_name": "Corvette C6 ZR1", "description": "6.2L supercharged V8 (LS9).", "engine_family": "V8 SC", "sort_order": 320},
    {"id": "corvette_c7_zr1", "display_name": "Corvette C7 ZR1", "description": "6.2L supercharged V8 (LT5).", "engine_family": "V8 SC", "sort_order": 330},
    {"id": "corvette_c7_z06", "display_name": "Corvette C7 Z06", "description": "6.2L supercharged V8 (LT4).", "engine_family": "V8 SC", "sort_order": 340},

    # ─────────────────────────────────────────────────────────────────────
    # v11 expansion (50 hypercars/supercars from audio-only auto-label).
    # ─────────────────────────────────────────────────────────────────────
    # Aston Martin
    {"id": "aston_dbs_superleggera", "display_name": "Aston Martin DBS Superleggera", "description": "5.2L twin-turbo V12.", "engine_family": "V12 TT", "sort_order": 400},
    {"id": "aston_martin_v12_vantage", "display_name": "Aston Martin V12 Vantage", "description": "5.2L twin-turbo V12.", "engine_family": "V12 TT", "sort_order": 405},
    {"id": "aston_martin_valkyrie", "display_name": "Aston Martin Valkyrie", "description": "Cosworth 6.5L NA V12 hybrid.", "engine_family": "V12 NA", "sort_order": 410},
    {"id": "aston_vanquish", "display_name": "Aston Martin Vanquish", "description": "5.2L twin-turbo V12.", "engine_family": "V12 TT", "sort_order": 415},

    # Ferrari classics
    {"id": "ferrari_360", "display_name": "Ferrari 360 Modena", "description": "3.6L NA flat-plane V8.", "engine_family": "V8 NA Flat", "sort_order": 420},
    {"id": "ferrari_enzo", "display_name": "Ferrari Enzo", "description": "6.0L NA V12.", "engine_family": "V12 NA", "sort_order": 425},
    {"id": "ferrari_f40", "display_name": "Ferrari F40", "description": "2.9L twin-turbo V8.", "engine_family": "V8 TT Flat", "sort_order": 430},
    {"id": "ferrari_f50", "display_name": "Ferrari F50", "description": "4.7L NA V12.", "engine_family": "V12 NA", "sort_order": 435},
    {"id": "ferrari_laferrari", "display_name": "Ferrari LaFerrari", "description": "6.3L NA V12 + hybrid.", "engine_family": "V12 NA", "sort_order": 440},

    # Lamborghini
    {"id": "lamborghini_aventador", "display_name": "Lamborghini Aventador", "description": "6.5L NA V12.", "engine_family": "V12 NA", "sort_order": 445},
    {"id": "lamborghini_diablo", "display_name": "Lamborghini Diablo", "description": "5.7-6.0L NA V12.", "engine_family": "V12 NA", "sort_order": 450},
    {"id": "lamborghini_gallardo", "display_name": "Lamborghini Gallardo", "description": "5.0/5.2L NA V10.", "engine_family": "V10 NA", "sort_order": 455},
    {"id": "lamborghini_murcielago", "display_name": "Lamborghini Murciélago", "description": "6.2-6.5L NA V12.", "engine_family": "V12 NA", "sort_order": 460},
    {"id": "lamborghini_revuelto", "display_name": "Lamborghini Revuelto", "description": "6.5L NA V12 + hybrid.", "engine_family": "V12 NA", "sort_order": 465},

    # McLaren
    {"id": "mclaren_f1", "display_name": "McLaren F1", "description": "BMW S70/2 6.1L NA V12.", "engine_family": "V12 NA", "sort_order": 470},
    {"id": "mclaren_mp4_12c", "display_name": "McLaren 12C", "description": "3.8L twin-turbo V8 (M838T).", "engine_family": "V8 TT Flat", "sort_order": 475},
    {"id": "mclaren_p1", "display_name": "McLaren P1", "description": "3.8L twin-turbo V8 + hybrid.", "engine_family": "V8 TT Flat", "sort_order": 480},
    {"id": "mclaren_speedtail", "display_name": "McLaren Speedtail", "description": "4.0L twin-turbo V8 + hybrid.", "engine_family": "V8 TT Flat", "sort_order": 485},
    {"id": "mclaren_artura", "display_name": "McLaren Artura", "description": "3.0L twin-turbo V6 hybrid.", "engine_family": "V6 TT Hybrid", "sort_order": 490},
    {"id": "mclaren_750s", "display_name": "McLaren 750S", "description": "4.0L twin-turbo V8 (M840T evo).", "engine_family": "V8 TT Flat", "sort_order": 495},

    # Pagani
    {"id": "pagani_huayra", "display_name": "Pagani Huayra", "description": "AMG-built 6.0L twin-turbo V12.", "engine_family": "V12 TT", "sort_order": 500},
    {"id": "pagani_zonda", "display_name": "Pagani Zonda", "description": "AMG 7.0-7.3L NA V12.", "engine_family": "V12 NA", "sort_order": 505},

    # Porsche
    {"id": "porsche_918_spyder", "display_name": "Porsche 918 Spyder", "description": "4.6L NA V8 + hybrid plug-in.", "engine_family": "V8 NA Flat", "sort_order": 510},
    {"id": "porsche_carrera_gt", "display_name": "Porsche Carrera GT", "description": "5.7L NA V10.", "engine_family": "V10 NA", "sort_order": 515},
    {"id": "porsche_911_turbo_s", "display_name": "Porsche 911 Turbo S", "description": "3.7-3.8L twin-turbo flat-6.", "engine_family": "Flat-6 TT", "sort_order": 520},
    {"id": "porsche_992_gt3_rs", "display_name": "Porsche 992 GT3 RS", "description": "4.0L NA flat-6, track-focused.", "engine_family": "Flat-6 NA", "sort_order": 525},
    {"id": "porsche_911_gt2_rs", "display_name": "Porsche 911 GT2 RS", "description": "3.8L twin-turbo flat-6.", "engine_family": "Flat-6 TT", "sort_order": 530},

    # Mercedes-AMG
    {"id": "mercedes_amg_one", "display_name": "Mercedes-AMG One", "description": "F1-derived 1.6L turbo V6 + 4 electric motors.", "engine_family": "V6 TT Hybrid", "sort_order": 535},
    {"id": "mercedes_amg_gt_black", "display_name": "Mercedes-AMG GT Black Series", "description": "4.0L twin-turbo flat-plane V8 (M178).", "engine_family": "V8 TT Flat", "sort_order": 540},
    {"id": "mercedes_sls_amg", "display_name": "Mercedes SLS AMG", "description": "6.2L NA V8 (M159).", "engine_family": "V8 NA Cross", "sort_order": 545},
    {"id": "mercedes_slr_mclaren", "display_name": "Mercedes SLR McLaren", "description": "5.4L supercharged V8.", "engine_family": "V8 SC", "sort_order": 550},

    # British
    {"id": "lotus_evija", "display_name": "Lotus Evija", "description": "All-electric hypercar.", "engine_family": "EV", "sort_order": 555},
    {"id": "jaguar_xj220", "display_name": "Jaguar XJ220", "description": "3.5L twin-turbo V6.", "engine_family": "V6 TT", "sort_order": 560},

    # Japanese
    {"id": "lexus_lfa", "display_name": "Lexus LFA", "description": "4.8L NA V10 9000rpm.", "engine_family": "V10 NA", "sort_order": 565},
    {"id": "nissan_gtr_r35", "display_name": "Nissan GT-R (R35)", "description": "3.8L twin-turbo V6 (VR38DETT).", "engine_family": "V6 TT", "sort_order": 570},
    {"id": "nissan_skyline_r34", "display_name": "Nissan Skyline GT-R (R34)", "description": "2.6L twin-turbo inline-6 (RB26DETT).", "engine_family": "I6 TT", "sort_order": 575},
    {"id": "acura_nsx_nc1", "display_name": "Honda/Acura NSX (NC1)", "description": "3.5L twin-turbo V6 hybrid.", "engine_family": "V6 TT Hybrid", "sort_order": 580},
    {"id": "mazda_rx7_fd", "display_name": "Mazda RX-7 FD", "description": "1.3L twin-rotor sequential turbo (13B-REW).", "engine_family": "Rotary Turbo", "sort_order": 585},

    # American
    {"id": "ford_gt_2017", "display_name": "Ford GT (2017)", "description": "3.5L twin-turbo V6 EcoBoost.", "engine_family": "V6 TT", "sort_order": 590},
    {"id": "ford_gt40", "display_name": "Ford GT40", "description": "NA V8 (vintage Le Mans).", "engine_family": "V8 NA Cross", "sort_order": 595},
    {"id": "dodge_viper_srt10", "display_name": "Dodge Viper SRT-10", "description": "8.4L NA V10.", "engine_family": "V10 NA", "sort_order": 600},
    {"id": "dodge_demon", "display_name": "Dodge Challenger SRT Demon", "description": "6.2L supercharged HEMI V8.", "engine_family": "V8 SC", "sort_order": 605},
    {"id": "corvette_z06_c8", "display_name": "Chevrolet Corvette C8 Z06", "description": "5.5L NA flat-plane V8 (LT6).", "engine_family": "V8 NA Flat", "sort_order": 610},

    # Hypercars
    {"id": "bugatti_chiron", "display_name": "Bugatti Chiron", "description": "8.0L quad-turbo W16.", "engine_family": "W16 TT", "sort_order": 615},
    {"id": "bugatti_veyron", "display_name": "Bugatti Veyron", "description": "8.0L quad-turbo W16.", "engine_family": "W16 TT", "sort_order": 620},
    {"id": "koenigsegg_jesko", "display_name": "Koenigsegg Jesko", "description": "5.0L twin-turbo V8.", "engine_family": "V8 TT Flat", "sort_order": 625},
    {"id": "koenigsegg_agera_rs", "display_name": "Koenigsegg Agera RS", "description": "5.0L twin-turbo V8.", "engine_family": "V8 TT Flat", "sort_order": 630},
    {"id": "koenigsegg_regera", "display_name": "Koenigsegg Regera", "description": "5.0L twin-turbo V8 + 3 electric motors.", "engine_family": "V8 TT Flat", "sort_order": 635},
    {"id": "hennessey_venom_f5", "display_name": "Hennessey Venom F5", "description": "6.6L twin-turbo cross-plane V8.", "engine_family": "V8 TT Cross", "sort_order": 640},
    {"id": "rimac_nevera", "display_name": "Rimac Nevera", "description": "All-electric hypercar (4 motors).", "engine_family": "EV", "sort_order": 645},

    # Always last.
    {
        "id": "other",
        "display_name": "Other / Unknown",
        "description": "Fallback when confidence is low or the engine isn't one we trained on.",
        "engine_family": None,
        "sort_order": 999,
    },
]
