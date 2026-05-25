"""Admin routes — triage user-submitted clips, promote them into new model classes.

Auth model: a single ADMIN_TOKEN env var. Send it as the bearer or as the
X-Admin-Token header. Don't expose any of these endpoints in the mobile app.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.config import settings
from app.db import get_session
from app.models import CarClass, Clip, Discovery

router = APIRouter(prefix="/admin", tags=["admin"])


def _check_admin(
    authorization: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    cfg = settings()
    if not cfg.admin_token:
        raise HTTPException(503, "admin disabled (no ADMIN_TOKEN set)")
    sent = x_admin_token
    if not sent and authorization and authorization.lower().startswith("bearer "):
        sent = authorization.split(None, 1)[1]
    if not sent or sent != cfg.admin_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad admin token")


admin_dep = Depends(_check_admin)


# ===================== Schemas =====================

class AdminClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    r2_key: str
    download_url: str
    duration_sec: float
    source: str
    review_status: str
    predicted_class_id: str | None
    predicted_confidence: float | None
    submitted_year: int | None
    submitted_make: str | None
    submitted_model: str | None
    submitted_trim: str | None
    submitted_notes: str | None
    user_correction_class_id: str | None
    created_at: datetime


class AdminClipPatch(BaseModel):
    review_status: Literal["pending", "accepted", "rejected"] | None = None
    user_correction_class_id: str | None = None  # set this to retag a clip


class AdminPendingSubmission(BaseModel):
    """Group of pending new-car-submission clips with the same submitted make/model."""
    submitted_year: int | None
    submitted_make: str
    submitted_model: str
    submitted_trim: str | None
    n_clips: int
    clip_ids: list[uuid.UUID]


class PromoteClassRequest(BaseModel):
    car_class_id: str = Field(..., min_length=1, max_length=64,
                              description="slug, e.g. 'ford_mustang_gt'")
    display_name: str = Field(..., min_length=1, max_length=120)
    description: str | None = None
    engine_family: str | None = None
    sort_order: int = 100
    clip_ids: list[uuid.UUID] = Field(default_factory=list,
                                       description="clips to retag onto this new class")


class PromoteClassResponse(BaseModel):
    car_class_id: str
    created: bool
    n_clips_retagged: int


# ===================== Routes =====================

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


@router.get("/clips", response_model=list[AdminClipOut], dependencies=[admin_dep])
async def list_clips(
    status_filter: str | None = Query(default=None, alias="status"),
    source: str | None = None,
    limit: int = Query(default=100, le=500),
    session: AsyncSession = Depends(get_session),
) -> list[AdminClipOut]:
    stmt = select(Clip).order_by(Clip.created_at.desc()).limit(limit)
    if status_filter:
        stmt = stmt.where(Clip.review_status == status_filter)
    if source:
        stmt = stmt.where(Clip.source == source)
    res = await session.execute(stmt)
    rows = res.scalars().all()
    out: list[AdminClipOut] = []
    for c in rows:
        out.append(AdminClipOut(
            id=c.id,
            user_id=c.user_id,
            r2_key=c.r2_key,
            download_url=storage.presigned_download_url(c.r2_key, expires_seconds=3600),
            duration_sec=c.duration_sec,
            source=c.source,
            review_status=c.review_status,
            predicted_class_id=c.predicted_class_id,
            predicted_confidence=c.predicted_confidence,
            submitted_year=c.submitted_year,
            submitted_make=c.submitted_make,
            submitted_model=c.submitted_model,
            submitted_trim=c.submitted_trim,
            submitted_notes=c.submitted_notes,
            user_correction_class_id=c.user_correction_class_id,
            created_at=c.created_at,
        ))
    return out


@router.patch("/clips/{clip_id}", response_model=AdminClipOut, dependencies=[admin_dep])
async def patch_clip(
    clip_id: uuid.UUID,
    body: AdminClipPatch,
    session: AsyncSession = Depends(get_session),
) -> AdminClipOut:
    clip = await session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    if body.review_status is not None:
        clip.review_status = body.review_status
    if body.user_correction_class_id is not None:
        # Verify the class exists
        klass = await session.get(CarClass, body.user_correction_class_id)
        if not klass:
            raise HTTPException(400, f"unknown car class: {body.user_correction_class_id}")
        clip.user_correction_class_id = body.user_correction_class_id
    await session.commit()
    await session.refresh(clip)
    return AdminClipOut(
        id=clip.id,
        user_id=clip.user_id,
        r2_key=clip.r2_key,
        download_url=storage.presigned_download_url(clip.r2_key, expires_seconds=3600),
        duration_sec=clip.duration_sec,
        source=clip.source,
        review_status=clip.review_status,
        predicted_class_id=clip.predicted_class_id,
        predicted_confidence=clip.predicted_confidence,
        submitted_year=clip.submitted_year,
        submitted_make=clip.submitted_make,
        submitted_model=clip.submitted_model,
        submitted_trim=clip.submitted_trim,
        submitted_notes=clip.submitted_notes,
        user_correction_class_id=clip.user_correction_class_id,
        created_at=clip.created_at,
    )


@router.get("/submissions", response_model=list[AdminPendingSubmission], dependencies=[admin_dep])
async def list_pending_submissions(
    session: AsyncSession = Depends(get_session),
) -> list[AdminPendingSubmission]:
    """Group new-car-submission clips by submitted make/model/year so you can see
    which candidate cars have enough audio to promote into a class."""
    res = await session.execute(
        select(Clip)
        .where(
            Clip.source == "new_car_submission",
            Clip.review_status == "pending",
            Clip.submitted_make.isnot(None),
            Clip.submitted_model.isnot(None),
        )
        .order_by(Clip.submitted_make, Clip.submitted_model, Clip.created_at)
    )
    grouped: dict[tuple, list[Clip]] = {}
    for c in res.scalars().all():
        key = (c.submitted_year, c.submitted_make, c.submitted_model, c.submitted_trim)
        grouped.setdefault(key, []).append(c)
    return [
        AdminPendingSubmission(
            submitted_year=year,
            submitted_make=make,
            submitted_model=model,
            submitted_trim=trim,
            n_clips=len(clips),
            clip_ids=[c.id for c in clips],
        )
        for (year, make, model, trim), clips in sorted(
            grouped.items(),
            key=lambda kv: (-len(kv[1]), kv[0][1] or "", kv[0][2] or ""),
        )
    ]


@router.post("/promote-class", response_model=PromoteClassResponse, dependencies=[admin_dep])
async def promote_class(
    body: PromoteClassRequest,
    session: AsyncSession = Depends(get_session),
) -> PromoteClassResponse:
    """Create a new CarClass row (or upsert) and retag a list of clips so the
    next training run sees them under that label.

    NOTE: this only updates the database. To actually train the model with
    the new class you still need to:
      1. Pull the audio for these clips down from R2.
      2. Drop them in data/raw/<car_class_id>/ and re-run carzam window/train.
      3. Upload the new checkpoint with apps/carzam-api/scripts/upload_model_to_r2.py.
      4. Restart the API container.
    """
    if not _SLUG_RE.match(body.car_class_id):
        raise HTTPException(400, "car_class_id must be lowercase ascii_with_underscores")

    stmt = pg_insert(CarClass).values(
        id=body.car_class_id,
        display_name=body.display_name,
        description=body.description,
        engine_family=body.engine_family,
        sort_order=body.sort_order,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "display_name": stmt.excluded.display_name,
            "description": stmt.excluded.description,
            "engine_family": stmt.excluded.engine_family,
            "sort_order": stmt.excluded.sort_order,
        },
    )
    await session.execute(stmt)

    # Determine if this was a fresh create vs an update
    was_existing = (await session.execute(
        select(func.count(Clip.id)).where(Clip.predicted_class_id == body.car_class_id)
    )).scalar_one() > 0

    n = 0
    if body.clip_ids:
        rows = (await session.execute(
            select(Clip).where(Clip.id.in_(body.clip_ids))
        )).scalars().all()
        for c in rows:
            c.user_correction_class_id = body.car_class_id
            c.review_status = "accepted"
            n += 1

    await session.commit()
    return PromoteClassResponse(
        car_class_id=body.car_class_id,
        created=not was_existing,
        n_clips_retagged=n,
    )


@router.get("/stats", dependencies=[admin_dep])
async def admin_stats(session: AsyncSession = Depends(get_session)) -> dict:
    """Quick health/data dashboard for ops."""
    n_users = (await session.execute(
        select(func.count()).select_from(__import__("app.models", fromlist=["User"]).User)
    )).scalar_one()
    n_clips = (await session.execute(select(func.count(Clip.id)))).scalar_one()
    n_pending = (await session.execute(
        select(func.count(Clip.id)).where(Clip.review_status == "pending")
    )).scalar_one()
    n_submissions = (await session.execute(
        select(func.count(Clip.id)).where(Clip.source == "new_car_submission")
    )).scalar_one()
    n_discoveries = (await session.execute(select(func.count()).select_from(Discovery))).scalar_one()
    return {
        "users": int(n_users),
        "clips": int(n_clips),
        "pending_review": int(n_pending),
        "new_car_submissions": int(n_submissions),
        "discoveries": int(n_discoveries),
    }
