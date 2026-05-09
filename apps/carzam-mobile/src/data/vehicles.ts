// Curated dataset for the make/model/trim/engine pickers.
// NOT exhaustive — covers ~50 makes that show up in enthusiast contexts.
// "Other" is always available as the last entry in any picker.

export const MAKES: readonly string[] = [
  // Mainstream
  "Acura", "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Bugatti",
  "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Ferrari", "Fiat",
  "Ford", "Genesis", "GMC", "Honda", "Hyundai", "Infiniti", "Jaguar", "Jeep",
  "Kia", "Koenigsegg", "Lamborghini", "Land Rover", "Lexus", "Lincoln",
  "Lotus", "Lucid", "Maserati", "Mazda", "McLaren", "Mercedes-AMG",
  "Mercedes-Benz", "Mini", "Mitsubishi", "Nissan", "Pagani", "Polestar",
  "Porsche", "Ram", "Rivian", "Rolls-Royce", "Subaru", "Suzuki", "Tesla",
  "Toyota", "Volkswagen", "Volvo",
  // Exotic / hypercar / boutique
  "Apollo", "Czinger", "Donkervoort", "Gordon Murray Automotive",
  "Hennessey", "Mosler", "Noble", "Pininfarina", "Radical", "Rimac",
  "Saleen", "SCG (Scuderia Cameron Glickenhaus)", "Spyker", "SSC",
  "TVR", "W Motors", "Zenvo",
  // Discontinued / classic enthusiast
  "AC", "De Tomaso", "Datsun", "DeLorean", "Fisker", "Hummer",
  "Isuzu", "Pontiac", "Saab", "Saturn", "Scion", "Smart",
] as const;

export const MODELS_BY_MAKE: Record<string, readonly string[]> = {
  Acura: ["Integra", "TLX", "TL", "RSX", "MDX", "NSX", "RDX", "ILX"],
  "Alfa Romeo": ["4C", "8C", "Giulia", "Stelvio", "Tonale", "Giulia Quadrifoglio"],
  "Aston Martin": [
    "DB11", "DB12", "DBS", "DBX", "Vantage", "V12 Vantage", "Valkyrie",
    "Valhalla", "Vanquish", "Rapide", "One-77", "Vulcan", "Victor",
  ],
  Audi: [
    "A3", "A4", "A5", "A6", "A7", "A8",
    "S3", "S4", "S5", "S6", "S7", "S8",
    "RS3", "RS4", "RS5", "RS6", "RS7", "RS Q8",
    "Q3", "Q5", "Q7", "Q8",
    "TT", "TTS", "TT RS", "R8", "e-tron GT", "RS e-tron GT",
  ],
  Bentley: ["Continental GT", "Flying Spur", "Bentayga", "Mulsanne"],
  BMW: [
    "1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "6 Series",
    "7 Series", "8 Series",
    "M2", "M3", "M4", "M5", "M6", "M8",
    "X1", "X2", "X3", "X4", "X5", "X5 M", "X6", "X6 M", "X7", "XM",
    "Z4", "i4", "i7", "i8", "iX",
  ],
  Bugatti: ["Chiron", "Veyron", "Divo", "Tourbillon"],
  Buick: ["Encore", "Enclave", "Envision", "Regal", "LaCrosse"],
  Cadillac: ["CT4", "CT4-V", "CT5", "CT5-V Blackwing", "CTS-V", "Escalade", "ATS", "ATS-V", "XT4", "XT5", "XT6"],
  Chevrolet: [
    "Camaro", "Camaro ZL1", "Camaro Z/28", "Corvette", "Corvette C5",
    "Corvette C6", "Corvette C7", "Corvette C8", "Corvette Z06",
    "Corvette ZR1", "Silverado", "Tahoe", "Suburban", "Trax", "Equinox",
    "Cruze", "Malibu", "Impala", "SS",
  ],
  Chrysler: ["300", "300 SRT", "Pacifica"],
  Dodge: [
    "Challenger", "Challenger Hellcat", "Challenger Demon", "Charger",
    "Charger Hellcat", "Charger Scat Pack", "Durango", "Durango SRT",
    "Viper", "Dart", "Magnum",
  ],
  Ferrari: [
    "458 Italia", "488 GTB", "488 Pista", "F8 Tributo", "812 Superfast",
    "812 GTS", "F12berlinetta", "599 GTB", "California", "Portofino",
    "Roma", "SF90 Stradale", "LaFerrari", "Purosangue", "F40", "F50",
    "Enzo", "458 Speciale",
  ],
  Fiat: ["500", "500X", "500L", "Abarth 500", "124 Spider"],
  Ford: [
    "Mustang", "Mustang GT", "Mustang Mach 1", "Mustang Dark Horse",
    "Mustang Shelby GT350", "Mustang Shelby GT500", "Mustang Mach-E",
    "F-150", "F-150 Raptor", "F-150 Lightning", "F-250", "F-350",
    "Ranger", "Bronco", "Bronco Raptor", "Explorer", "Edge", "Escape",
    "Fiesta", "Fiesta ST", "Focus", "Focus RS", "Focus ST", "Fusion", "GT",
  ],
  Genesis: ["G70", "G80", "G90", "GV60", "GV70", "GV80"],
  GMC: ["Sierra", "Yukon", "Acadia", "Terrain", "Hummer EV"],
  Honda: [
    "Accord", "Accord Sport", "Civic", "Civic Si", "Civic Type R", "CR-V",
    "Fit", "HR-V", "Odyssey", "Passport", "Pilot", "Prelude", "Ridgeline",
    "S2000", "Insight", "Integra Type R",
  ],
  Hyundai: [
    "Elantra", "Elantra N", "Sonata", "Sonata N Line", "Veloster",
    "Veloster N", "Kona", "Kona N", "Tucson", "Santa Fe", "Palisade",
    "Ioniq 5", "Ioniq 5 N", "Ioniq 6", "Genesis Coupe",
  ],
  Infiniti: ["G35", "G37", "Q50", "Q60", "QX50", "QX60", "QX80"],
  Jaguar: ["F-Type", "F-Pace", "I-Pace", "XE", "XF", "XJ", "Project 8"],
  Jeep: ["Wrangler", "Wrangler Rubicon", "Grand Cherokee", "Grand Cherokee SRT", "Grand Cherokee Trackhawk", "Compass", "Cherokee", "Gladiator"],
  Kia: ["Forte", "Forte GT", "K5", "K5 GT", "Stinger", "Stinger GT", "EV6", "EV6 GT", "Telluride", "Sorento", "Sportage", "Soul"],
  Koenigsegg: ["Agera", "Jesko", "Regera", "CCX", "Gemera"],
  Lamborghini: [
    "Aventador", "Aventador SVJ", "Huracán", "Huracán Performante",
    "Huracán STO", "Huracán Tecnica", "Revuelto", "Urus", "Gallardo",
    "Murciélago", "Diablo", "Countach",
  ],
  "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport", "Range Rover Velar", "Range Rover Evoque", "Discovery Sport"],
  Lexus: [
    "IS", "IS 350", "IS 500", "ES", "GS", "GS F", "LS", "RC", "RC F",
    "LC", "LC 500", "LC 500h", "LFA", "LFA Nürburgring", "RX", "GX", "LX",
    "NX", "UX",
  ],
  Lincoln: ["Aviator", "Navigator", "Nautilus", "Corsair", "MKZ"],
  Lotus: ["Elise", "Exige", "Evora", "Emira", "Evija", "Eletre", "Esprit", "3-Eleven"],
  Lucid: ["Air", "Air Sapphire", "Gravity"],
  Maserati: ["Ghibli", "Quattroporte", "Levante", "GranTurismo", "GranCabrio", "MC20", "MC20 Cielo", "Granturismo MC", "MC12"],
  Mazda: ["MX-5 Miata", "RX-7", "RX-8", "Mazda3", "Mazdaspeed3", "Mazda6", "Mazdaspeed6", "CX-5", "CX-9", "CX-30", "CX-50", "CX-90"],
  McLaren: [
    "540C", "570S", "570GT", "600LT", "620R", "650S", "675LT", "720S",
    "750S", "765LT", "Artura", "GT", "P1", "P1 GTR", "Senna", "Senna GTR",
    "Speedtail", "Elva", "F1", "F1 LM", "F1 GTR", "Solus GT", "W1",
  ],
  "Mercedes-AMG": [
    "A45", "C43", "C63", "C63 S", "CLA45", "E53", "E63", "E63 S",
    "GLC43", "GLC63", "GLE53", "GLE63", "GT", "GT R", "GT 4-Door",
    "GT Black Series", "S63", "SL55", "SL63", "ONE", "Project One",
    "SLS AMG", "SLR McLaren",
  ],
  "Mercedes-Benz": [
    "A-Class", "C-Class", "E-Class", "S-Class", "CLA", "CLS", "GLA",
    "GLC", "GLE", "GLS", "G-Class", "G63 AMG", "SL", "SLK", "SLS AMG",
    "EQS", "EQE",
  ],
  Mini: ["Cooper", "Cooper S", "John Cooper Works", "Countryman", "Clubman"],
  Mitsubishi: ["Lancer", "Lancer Evolution", "Eclipse", "Eclipse Cross", "Outlander", "Outlander Sport", "3000GT"],
  Nissan: [
    "Altima", "Maxima", "Sentra", "Sentra SE-R", "240SX", "300ZX",
    "350Z", "370Z", "Z (RZ34)", "GT-R", "GT-R Nismo", "Skyline GT-R",
    "Pathfinder", "Murano", "Rogue", "Frontier", "Titan", "Versa", "Leaf",
    "Ariya", "Silvia", "Juke", "Juke Nismo",
  ],
  Pagani: ["Zonda", "Huayra", "Utopia"],
  Polestar: ["Polestar 1", "Polestar 2", "Polestar 3", "Polestar 4"],
  Porsche: [
    "911", "911 Carrera", "911 Carrera S", "911 GT3", "911 GT3 RS",
    "911 GT2 RS", "911 Turbo", "911 Turbo S", "718 Cayman",
    "718 Cayman S", "718 Cayman GT4", "718 Cayman GT4 RS",
    "718 Boxster", "718 Spyder", "Panamera", "Panamera Turbo",
    "Cayenne", "Cayenne Turbo", "Macan", "Taycan", "Taycan Turbo",
  ],
  Ram: ["1500", "1500 TRX", "2500", "3500", "ProMaster"],
  Rivian: ["R1T", "R1S", "R3"],
  "Rolls-Royce": ["Phantom", "Ghost", "Wraith", "Dawn", "Cullinan", "Spectre"],
  Subaru: [
    "WRX", "WRX STI", "STI", "Impreza", "Legacy", "Outback", "Forester",
    "BRZ", "BRZ tS", "Crosstrek", "Ascent",
  ],
  Suzuki: ["Swift", "Swift Sport", "Jimny", "Vitara", "SX4"],
  Tesla: ["Model S", "Model S Plaid", "Model 3", "Model 3 Performance", "Model X", "Model Y", "Cybertruck", "Roadster"],
  Toyota: [
    "86", "GR86", "Camry", "Camry TRD", "Corolla", "GR Corolla",
    "Corolla GR", "Crown", "Mirai", "Prius", "Prius Prime", "RAV4",
    "RAV4 Prime", "Highlander", "4Runner", "Tacoma", "Tundra", "Sequoia",
    "Land Cruiser", "Supra", "GR Supra", "MR2", "Celica", "Celica GT-Four",
  ],
  Volkswagen: ["Golf", "Golf GTI", "Golf R", "Jetta", "Jetta GLI", "Passat", "Tiguan", "Atlas", "Arteon", "ID.4", "ID. Buzz", "Beetle"],
  Volvo: ["S60", "S60 Polestar", "S90", "V60", "XC40", "XC60", "XC90", "EX30"],

  // Exotic / hypercar / boutique
  Apollo: ["IE", "Project EVO", "Arrow"],
  Czinger: ["21C"],
  Donkervoort: ["D8 GTO", "F22"],
  "Gordon Murray Automotive": ["T.50", "T.50s Niki Lauda", "T.33"],
  Hennessey: ["Venom GT", "Venom F5", "Venom F5 Roadster"],
  Mosler: ["MT900", "Photon"],
  Noble: ["M600", "M400", "M12"],
  Pininfarina: ["Battista"],
  Radical: ["SR1", "SR3", "SR8", "SR10", "RXC"],
  Rimac: ["Nevera", "Concept_One"],
  Saleen: ["S1", "S7", "S7 Twin Turbo", "S281", "Mustang"],
  "SCG (Scuderia Cameron Glickenhaus)": ["SCG 003", "SCG 004S", "Boot"],
  Spyker: ["C8", "C8 Aileron", "C12 Zagato"],
  SSC: ["Tuatara", "Ultimate Aero"],
  TVR: ["Cerbera", "Tuscan", "Sagaris", "Griffith"],
  "W Motors": ["Lykan HyperSport", "Fenyr SuperSport"],
  Zenvo: ["TSR-S", "TSR-GT", "ST1", "Aurora"],

  // Discontinued / classic enthusiast
  AC: ["Cobra", "Ace", "428"],
  "De Tomaso": ["Pantera", "Mangusta", "P72", "Vallelunga"],
  Datsun: ["240Z", "260Z", "280Z", "510", "Roadster"],
  DeLorean: ["DMC-12"],
  Fisker: ["Karma", "Ocean", "Pear"],
  Hummer: ["H1", "H2", "H3"],
  Isuzu: ["Trooper", "Rodeo", "VehiCROSS"],
  Pontiac: ["Firebird", "Trans Am", "GTO", "G8", "Solstice"],
  Saab: ["9-3", "9-5", "900", "9000"],
  Saturn: ["Sky", "Sky Redline"],
  Scion: ["FR-S", "tC", "xB", "xD"],
  Smart: ["Fortwo", "Forfour"],
};

export const ENGINES: readonly string[] = [
  "I3 NA", "I3 Turbo",
  "I4 NA", "I4 Turbo", "I4 Twin Turbo",
  "I5 NA", "I5 Turbo",
  "I6 NA", "I6 Turbo", "I6 Twin Turbo",
  "V6 NA", "V6 Turbo", "V6 Twin Turbo", "V6 Supercharged",
  "V8 NA", "V8 Turbo", "V8 Twin Turbo", "V8 Supercharged",
  "V10 NA", "V10 Twin Turbo",
  "V12 NA", "V12 Twin Turbo",
  "Flat-4 NA", "Flat-4 Turbo",
  "Flat-6 NA", "Flat-6 Turbo", "Flat-6 Twin Turbo",
  "Rotary",
  "Hybrid (V6 + electric)", "Hybrid (V8 + electric)", "Hybrid (V12 + electric)",
  "Electric (single motor)", "Electric (dual motor)", "Electric (tri motor)",
];

/**
 * Returns the model list for a given make. Returns [] if the make is "Other"
 * or unknown — the picker should fall back to free-text in that case.
 */
export function modelsFor(make: string | null): readonly string[] {
  if (!make) return [];
  return MODELS_BY_MAKE[make] ?? [];
}
