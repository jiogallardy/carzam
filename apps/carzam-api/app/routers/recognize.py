"""The Carzam button — record → upload → classify."""
import uuid
from io import BytesIO

import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app import inference, storage
from app.config import settings
from app.db import get_session
from app.models import CarClass, Clip, Discovery, User
from app.schemas import (
    AggregateRecognizeResponse,
    ClipOut,
    RecognizeResponse,
    SubmitNewCarRequest,
    TopCandidate,
)
from app.security import current_user

router = APIRouter(tags=["recognize"])


def _wav_duration(wav_bytes: bytes) -> float:
    info = sf.info(BytesIO(wav_bytes))
    return float(info.frames) / float(info.samplerate)


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    audio: UploadFile = File(...),
    model: str | None = Form(default=None),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> RecognizeResponse:
    wav_bytes = await audio.read()
    if not wav_bytes:
        raise HTTPException(400, "empty body")

    try:
        duration = _wav_duration(wav_bytes)
    except Exception as e:
        raise HTTPException(400, f"cannot decode wav: {e}") from e

    clip_id = uuid.uuid4()
    r2_key = f"clips/{user.id}/{clip_id}.wav"
    storage.put_clip(r2_key, wav_bytes)

    try:
        result = inference.predict_bytes(wav_bytes, model_id=model)
    except FileNotFoundError as e:
        raise HTTPException(404, f"unknown model: {model}") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    clip = Clip(
        id=clip_id,
        user_id=user.id,
        r2_key=r2_key,
        duration_sec=duration,
        source="recognize",
        predicted_class_id=result["predicted_class_id"],
        predicted_confidence=result["predicted_confidence"],
        predicted_state=result["predicted_state"],
        all_probs=result["all_probs"],
        model_run_id=result["model_run_id"],
    )
    session.add(clip)

    # Build display name lookup for top-3
    cars_in_top3 = [c for c, _ in result["top3"]]
    classes_res = await session.execute(select(CarClass).where(CarClass.id.in_(cars_in_top3)))
    by_id = {c.id: c for c in classes_res.scalars().all()}
    top3 = [
        TopCandidate(
            car_class_id=cid,
            display_name=(by_id[cid].display_name if cid in by_id else cid),
            confidence=conf,
        )
        for cid, conf in result["top3"]
    ]

    # Upsert discovery if matched (and not 'other')
    discovery_unlocked = False
    if result["predicted_class_id"] and result["predicted_class_id"] != "other":
        # Try insert; if exists, update counters
        existing = (await session.execute(
            select(Discovery).where(
                Discovery.user_id == user.id,
                Discovery.car_class_id == result["predicted_class_id"],
            )
        )).scalar_one_or_none()
        if existing is None:
            session.add(Discovery(
                user_id=user.id,
                car_class_id=result["predicted_class_id"],
                first_clip_id=clip_id,
                total_matches=1,
                best_confidence=result["predicted_confidence"],
            ))
            discovery_unlocked = True
        else:
            existing.total_matches += 1
            if result["predicted_confidence"] > existing.best_confidence:
                existing.best_confidence = result["predicted_confidence"]

    # Recompute completion % for response
    n_total = (await session.execute(
        select(func.count(CarClass.id)).where(CarClass.id != "other")
    )).scalar_one()
    n_discovered = (await session.execute(
        select(func.count(Discovery.car_class_id))
        .where(Discovery.user_id == user.id, Discovery.car_class_id != "other")
    )).scalar_one()
    if discovery_unlocked:
        n_discovered = int(n_discovered) + 1  # the row we just added isn't counted yet pre-commit
    pct = (100.0 * n_discovered / n_total) if n_total else 0.0

    await session.commit()

    matched = result["predicted_class_id"]
    matched_display = by_id[matched].display_name if matched and matched in by_id else None

    return RecognizeResponse(
        clip_id=clip_id,
        predicted_class_id=matched,
        predicted_display_name=matched_display,
        confidence=result["predicted_confidence"],
        predicted_state=result["predicted_state"],
        top3=top3,
        discovery_unlocked=discovery_unlocked,
        completion_percent=round(pct, 1),
        n_discovered=int(n_discovered),
        n_total=int(n_total),
        silence=bool(result.get("silence", False)),
        model_id=str(result.get("model_id", "default")),
    )


@router.post("/recognize/aggregate", response_model=AggregateRecognizeResponse)
async def recognize_aggregate(
    audio: UploadFile = File(...),
    model: str | None = Form(default=None),
    commit: bool = Form(default=False),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> AggregateRecognizeResponse:
    """Shazam-style sliding-window inference.

    The mobile app polls this endpoint with a growing audio buffer
    (every ~2s, up to ~15s). Server averages softmax across 5s windows
    at 2.5s hop and returns a verdict the client uses to decide whether
    to commit or keep listening.

    `commit=False` (default): pure inference, no R2 upload, no DB write.
    `commit=True`: save the clip to R2 and update discovery — used by the
    mobile app once it's confident.
    """
    wav_bytes = await audio.read()
    if not wav_bytes:
        raise HTTPException(400, "empty body")

    try:
        duration = _wav_duration(wav_bytes)
    except Exception as e:
        raise HTTPException(400, f"cannot decode wav: {e}") from e

    try:
        result = inference.predict_bytes_aggregate(wav_bytes, model_id=model)
    except FileNotFoundError as e:
        raise HTTPException(404, f"unknown model: {model}") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    # Build a display-name lookup covering top-3 + the top-1 prediction.
    needed_ids = {cid for cid, _ in result["top3"]}
    if result["predicted_class_id"]:
        needed_ids.add(result["predicted_class_id"])
    classes_res = await session.execute(
        select(CarClass).where(CarClass.id.in_(needed_ids))
    ) if needed_ids else None
    by_id = {c.id: c for c in classes_res.scalars().all()} if classes_res else {}

    top3 = [
        TopCandidate(
            car_class_id=cid,
            display_name=by_id[cid].display_name if cid in by_id else cid,
            confidence=conf,
        )
        for cid, conf in result["top3"]
    ]
    matched = result["predicted_class_id"]
    matched_display = by_id[matched].display_name if matched and matched in by_id else None

    # No commit -> stateless lightweight response. This is the path the
    # mobile UI polls during the "listening..." phase.
    if not commit:
        return AggregateRecognizeResponse(
            predicted_class_id=matched,
            predicted_display_name=matched_display,
            confidence=result["predicted_confidence"],
            is_confident=result["is_confident"],
            couldnt_place=result["couldnt_place"],
            n_windows_aggregated=result["n_windows_aggregated"],
            duration_seconds=result["duration_seconds"],
            top3=top3,
            silence=bool(result.get("silence", False)),
            model_id=str(result.get("model_id", "default")),
        )

    # commit=True -> save the clip + update discovery so this counts
    # toward the user's collection.
    clip_id = uuid.uuid4()
    r2_key = f"clips/{user.id}/{clip_id}.wav"
    storage.put_clip(r2_key, wav_bytes)

    clip = Clip(
        id=clip_id,
        user_id=user.id,
        r2_key=r2_key,
        duration_sec=duration,
        source="recognize",
        predicted_class_id=matched,
        predicted_confidence=result["predicted_confidence"],
        predicted_state="idle",   # aggregate path doesn't track state
        all_probs={cid: conf for cid, conf in result["top3"]},  # store top-3 only — full set would bloat row
        model_run_id=settings().model_run_id,
    )
    session.add(clip)

    discovery_unlocked = False
    if matched and matched != "other":
        existing = (await session.execute(
            select(Discovery).where(
                Discovery.user_id == user.id,
                Discovery.car_class_id == matched,
            )
        )).scalar_one_or_none()
        if existing is None:
            session.add(Discovery(
                user_id=user.id,
                car_class_id=matched,
                first_clip_id=clip_id,
                total_matches=1,
                best_confidence=result["predicted_confidence"],
            ))
            discovery_unlocked = True
        else:
            existing.total_matches += 1
            if result["predicted_confidence"] > existing.best_confidence:
                existing.best_confidence = result["predicted_confidence"]

    n_total = (await session.execute(
        select(func.count(CarClass.id)).where(CarClass.id != "other")
    )).scalar_one()
    n_discovered = (await session.execute(
        select(func.count(Discovery.car_class_id))
        .where(Discovery.user_id == user.id, Discovery.car_class_id != "other")
    )).scalar_one()
    if discovery_unlocked:
        n_discovered = int(n_discovered) + 1
    pct = (100.0 * n_discovered / n_total) if n_total else 0.0

    await session.commit()

    return AggregateRecognizeResponse(
        predicted_class_id=matched,
        predicted_display_name=matched_display,
        confidence=result["predicted_confidence"],
        is_confident=result["is_confident"],
        couldnt_place=result["couldnt_place"],
        n_windows_aggregated=result["n_windows_aggregated"],
        duration_seconds=result["duration_seconds"],
        top3=top3,
        silence=bool(result.get("silence", False)),
        model_id=str(result.get("model_id", "default")),
        clip_id=clip_id,
        discovery_unlocked=discovery_unlocked,
        completion_percent=round(pct, 1),
        n_discovered=int(n_discovered),
        n_total=int(n_total),
    )


@router.post("/clips/{clip_id}/submit-as-new-car", response_model=ClipOut)
async def submit_as_new_car(
    clip_id: uuid.UUID,
    body: SubmitNewCarRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ClipOut:
    clip = await session.get(Clip, clip_id)
    if clip is None or clip.user_id != user.id:
        raise HTTPException(404, "clip not found")
    clip.source = "new_car_submission"
    clip.submitted_year = body.year
    clip.submitted_make = body.make
    clip.submitted_model = body.model
    clip.submitted_trim = body.trim
    clip.submitted_notes = body.notes
    await session.commit()
    await session.refresh(clip)
    return ClipOut.model_validate(clip)


@router.post("/me/cars/{car_id}/clips", response_model=ClipOut, status_code=201)
async def upload_my_car_clip(
    car_id: uuid.UUID,
    audio: UploadFile = File(...),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ClipOut:
    """Upload an audio recording of a car you own. No prediction is run; this is reference data."""
    from app.models import OwnedCar
    car = await session.get(OwnedCar, car_id)
    if car is None or car.user_id != user.id:
        raise HTTPException(404, "car not found")
    wav = await audio.read()
    if not wav:
        raise HTTPException(400, "empty body")
    try:
        duration = _wav_duration(wav)
    except Exception as e:
        raise HTTPException(400, f"cannot decode wav: {e}") from e
    clip_id = uuid.uuid4()
    r2_key = f"my-cars/{user.id}/{car_id}/{clip_id}.wav"
    storage.put_clip(r2_key, wav)
    clip = Clip(
        id=clip_id,
        user_id=user.id,
        r2_key=r2_key,
        duration_sec=duration,
        source="my_car",
        owned_car_id=car_id,
    )
    session.add(clip)
    await session.commit()
    await session.refresh(clip)
    return ClipOut.model_validate(clip)
