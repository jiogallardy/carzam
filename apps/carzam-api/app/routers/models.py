"""Lists model checkpoints available in R2 so the mobile app can pick one."""
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import inference, storage
from app.db import get_session
from app.models import CarClass, User
from app.schemas import CarClassOut, ModelInfo, ModelsResponse
from app.security import current_user

router = APIRouter(tags=["models"])


def _classes_r2_key(model_id: str) -> str:
    """Where this model's classes.json lives in R2."""
    if model_id == inference.LEGACY_DEFAULT_ID:
        from app.config import settings
        return settings().classes_r2_key
    return f"model/{model_id}/classes.json"


@router.get("/models", response_model=ModelsResponse)
async def list_models(_user: User = Depends(current_user)) -> ModelsResponse:
    ids = inference.list_available()
    default = inference.default_id()
    return ModelsResponse(
        default_id=default,
        models=[
            ModelInfo(
                id=mid,
                label=mid if mid != "default" else "default (legacy)",
                is_default=(mid == default),
                is_loaded=inference.is_loaded(mid),
            )
            for mid in ids
        ],
    )


@router.get("/models/{model_id}/classes", response_model=list[CarClassOut])
async def list_model_classes(
    model_id: str,
    _user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> list[CarClassOut]:
    """Return the car classes this specific model can identify.

    Reads model/<id>/classes.json from R2 (it's only ~2 KB), then joins
    each class id against the CarClass DB so the response carries the
    full pretty metadata (display_name, image_url, engine_family).

    Unknown ids that aren't in the DB still come back with the raw id as
    `display_name` so the mobile UI never silently drops them.
    """
    key = _classes_r2_key(model_id)
    if not storage.object_exists(key):
        raise HTTPException(404, f"unknown model: {model_id!r}")

    try:
        body = storage.get_clip(key)
        payload = json.loads(body)
    except Exception as e:
        raise HTTPException(502, f"could not read model classes: {e}") from e

    cars = payload.get("cars") or []
    if not cars:
        return []

    # Pull all matching CarClass rows in one query.
    db_res = await session.execute(select(CarClass).where(CarClass.id.in_(cars)))
    db_rows = {c.id: c for c in db_res.scalars().all()}

    out: list[CarClassOut] = []
    for idx, cid in enumerate(cars):
        if cid in db_rows:
            out.append(CarClassOut.model_validate(db_rows[cid]))
        else:
            # Unknown id (e.g. legacy specialist labels): synthesize a row
            # so the mobile UI still has something to render.
            out.append(CarClassOut(
                id=cid,
                display_name=cid.replace("_", " ").title(),
                description=None,
                engine_family=None,
                image_url=f"/car-classes/{cid}/image",
                sort_order=1000 + idx,
            ))
    return out
