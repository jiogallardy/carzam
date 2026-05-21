// Mirror of FastAPI schemas in apps/carzam-api/app/schemas.py.
// Keep these in sync; if it gets out of hand later we can codegen from OpenAPI.

export type User = {
  id: string;
  email: string | null;
  display_name: string | null;
  created_at: string;
};

export type AuthResponse = { jwt: string; user: User };

export type OwnedCar = {
  id: string;
  year: number;
  make: string;
  model: string;
  trim: string | null;
  color: string | null;
  mods_notes: string | null;
  created_at: string;
};

export type OwnedCarCreate = {
  year: number;
  make: string;
  model: string;
  trim?: string | null;
  color?: string | null;
  mods_notes?: string | null;
};

export type TopCandidate = {
  car_class_id: string;
  display_name: string;
  confidence: number;
};

export type RecognizeResponse = {
  clip_id: string;
  predicted_class_id: string | null;
  predicted_display_name: string | null;
  confidence: number;
  predicted_state: string;
  top3: TopCandidate[];
  discovery_unlocked: boolean;
  completion_percent: number;
  n_discovered: number;
  n_total: number;
  silence?: boolean;
  model_id?: string;
};

// Shazam-style aggregated response from /recognize/aggregate.
// `is_confident` and `couldnt_place` drive the mobile commit decision.
// Discovery-related fields are only populated when commit=true.
export type AggregateRecognizeResponse = {
  predicted_class_id: string | null;
  predicted_display_name: string | null;
  confidence: number;
  is_confident: boolean;
  couldnt_place: boolean;
  n_windows_aggregated: number;
  duration_seconds: number;
  top3: TopCandidate[];
  silence?: boolean;
  model_id?: string;
  clip_id?: string | null;
  discovery_unlocked?: boolean;
  completion_percent?: number | null;
  n_discovered?: number | null;
  n_total?: number | null;
};

export type ModelInfo = {
  id: string;
  label: string;
  is_default: boolean;
  is_loaded: boolean;
};

export type ModelsResponse = {
  default_id: string;
  models: ModelInfo[];
};

export type CarClass = {
  id: string;
  display_name: string;
  description: string | null;
  engine_family: string | null;
  image_url: string | null;
  sort_order: number;
};

export type Discovery = {
  car_class: CarClass;
  first_discovered_at: string;
  total_matches: number;
  best_confidence: number;
};

export type Stats = {
  n_discovered: number;
  n_total: number;
  completion_percent: number;
  total_recognitions: number;
};

export type SubmitNewCarRequest = {
  year: number;
  make: string;
  model: string;
  trim?: string | null;
  notes?: string | null;
};

export type ClipOut = {
  id: string;
  source: string;
  predicted_class_id: string | null;
  predicted_confidence: number | null;
  predicted_state: string | null;
  created_at: string;
};
