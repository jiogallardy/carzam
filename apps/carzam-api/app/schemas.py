"""Pydantic schemas — API request/response shapes. Distinct from DB models so we control what's exposed."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ============= Auth =============

class GoogleSignInRequest(BaseModel):
    id_token: str


class AppleSignInRequest(BaseModel):
    id_token: str
    authorization_code: str | None = None
    full_name: str | None = None  # Apple gives this only on first sign-in
    email: str | None = None


class AuthResponse(BaseModel):
    jwt: str
    user: "UserOut"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: str | None
    display_name: str | None
    created_at: datetime


# ============= Cars =============

class OwnedCarCreate(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    make: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=64)
    trim: str | None = Field(default=None, max_length=64)
    color: str | None = Field(default=None, max_length=48)
    mods_notes: str | None = None


class OwnedCarOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    year: int
    make: str
    model: str
    trim: str | None
    color: str | None
    mods_notes: str | None
    created_at: datetime


# ============= Recognize =============

class TopCandidate(BaseModel):
    car_class_id: str
    display_name: str
    confidence: float


class RecognizeResponse(BaseModel):
    clip_id: uuid.UUID
    predicted_class_id: str | None
    predicted_display_name: str | None
    confidence: float
    predicted_state: str
    top3: list[TopCandidate]
    discovery_unlocked: bool
    completion_percent: float
    n_discovered: int
    n_total: int
    silence: bool = False
    model_id: str = "default"


class AggregateRecognizeResponse(BaseModel):
    """Shazam-style aggregated prediction across a sliding window.

    The mobile app polls this endpoint with a growing audio buffer
    (every 2s, capped at ~15s). It commits the result when
    `is_confident` flips true on two consecutive polls; if
    `couldnt_place` stays true through the cap, it shows the friendly
    "Couldn't quite place that one" message.
    """
    predicted_class_id: str | None
    predicted_display_name: str | None
    confidence: float
    # True when confidence >= 0.60 — the mobile may commit on stable hit.
    is_confident: bool
    # True when confidence < 0.40 — surface the friendly fallback copy.
    couldnt_place: bool
    # How many 5s windows the server actually scored. Useful for the
    # client to know if its buffer is long enough to trust.
    n_windows_aggregated: int
    duration_seconds: float
    top3: list[TopCandidate]
    silence: bool = False
    model_id: str = "default"
    # Only set when commit=true: a real clip was saved + a discovery
    # was processed. Identical shape to RecognizeResponse's fields.
    clip_id: uuid.UUID | None = None
    discovery_unlocked: bool = False
    completion_percent: float | None = None
    n_discovered: int | None = None
    n_total: int | None = None


class ModelInfo(BaseModel):
    id: str
    label: str
    is_default: bool
    is_loaded: bool


class ModelsResponse(BaseModel):
    default_id: str
    models: list[ModelInfo]


class SubmitNewCarRequest(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    make: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=64)
    trim: str | None = Field(default=None, max_length=64)
    notes: str | None = None


# ============= Discovered / Stats =============

class CarClassOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    display_name: str
    description: str | None
    engine_family: str | None
    image_url: str | None
    sort_order: int


class DiscoveryOut(BaseModel):
    car_class: CarClassOut
    first_discovered_at: datetime
    total_matches: int
    best_confidence: float


class StatsOut(BaseModel):
    n_discovered: int
    n_total: int
    completion_percent: float
    total_recognitions: int


# ============= Clips =============

class ClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    source: str
    predicted_class_id: str | None
    predicted_confidence: float | None
    predicted_state: str | None
    created_at: datetime


AuthResponse.model_rebuild()
