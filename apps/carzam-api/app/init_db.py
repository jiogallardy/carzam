"""Create tables + seed car_classes. Idempotent; safe to run on every startup or manually."""
from __future__ import annotations

import asyncio

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Base, _session_factory, engine
from app.models import CarClass  # noqa: F401  ensure registered
from app.models import Clip, Discovery, OwnedCar, User  # noqa: F401
from app.seeds import CAR_CLASS_SEEDS


async def create_tables() -> None:
    async with engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _derive_image_url(seed: dict) -> str | None:
    """Default image_url to the /car-classes/<id>/image proxy route unless
    the seed explicitly provided one. 'other' has no real image — null it.
    """
    if "image_url" in seed:
        return seed["image_url"]
    car_id = seed["id"]
    if car_id == "other":
        return None
    return f"/car-classes/{car_id}/image"


async def seed_car_classes(session: AsyncSession) -> None:
    """Upsert car_classes from seeds. Updates display_name/description/
    engine_family/sort_order/image_url on conflict."""
    for seed in CAR_CLASS_SEEDS:
        values = {**seed, "image_url": _derive_image_url(seed)}
        stmt = pg_insert(CarClass).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "display_name": stmt.excluded.display_name,
                "description": stmt.excluded.description,
                "engine_family": stmt.excluded.engine_family,
                "sort_order": stmt.excluded.sort_order,
                "image_url": stmt.excluded.image_url,
            },
        )
        await session.execute(stmt)
    await session.commit()


async def init() -> None:
    await create_tables()
    async with _session_factory() as session:
        await seed_car_classes(session)


if __name__ == "__main__":
    asyncio.run(init())
