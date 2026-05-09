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
