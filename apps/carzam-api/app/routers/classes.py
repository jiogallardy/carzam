"""Read-only access to the model's class set."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.config import settings
from app.db import get_session
from app.models import CarClass
from app.schemas import CarClassOut

router = APIRouter(tags=["classes"])


# Tiny on-disk LRU for car images so we don't hammer R2 on every request.
# Each image is ~80-150KB; 72 cars => ~10MB resident. Cheap.
_IMAGE_CACHE_DIR = Path("/data/car_images")


@router.get("/car-classes", response_model=list[CarClassOut])
async def list_car_classes(session: AsyncSession = Depends(get_session)) -> list[CarClassOut]:
    res = await session.execute(
        select(CarClass).where(CarClass.id != "other").order_by(CarClass.sort_order, CarClass.id)
    )
    return [CarClassOut.model_validate(c) for c in res.scalars().all()]


@router.get("/car-classes/{car_id}/image")
async def get_car_class_image(
    car_id: str,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Proxy the per-class image from R2.

    Images live at car_images/<car_id>.jpg in the R2 bucket — uploaded
    via apps/carzam-api/scripts/upload_car_images.py. We cache locally
    in /data/car_images to avoid hammering R2 on every mobile request.
    Falls back to 404 if the class doesn't exist or has no image.
    """
    # Confirm the class is real — defensive 404 instead of leaking S3 errors.
    res = await session.execute(select(CarClass).where(CarClass.id == car_id))
    if res.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail=f"unknown car_id: {car_id!r}")

    _IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _IMAGE_CACHE_DIR / f"{car_id}.jpg"
    r2_key = f"car_images/{car_id}.jpg"

    # Hot-path: serve from local cache.
    if local_path.exists():
        body = local_path.read_bytes()
    else:
        # Cold: pull from R2, cache to disk, serve.
        if not storage.object_exists(r2_key):
            raise HTTPException(status_code=404, detail=f"no image for car_id: {car_id!r}")
        body = storage.get_clip(r2_key)
        local_path.write_bytes(body)

    return Response(
        content=body,
        media_type="image/jpeg",
        headers={
            # 30 days — refresh when seeds bump the URL (which they don't,
            # but the iOS layer can bust by adding ?v=... if needed).
            "Cache-Control": "public, max-age=2592000, immutable",
        },
    )
