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


async def seed_car_classes(session: AsyncSession) -> None:
    """Upsert car_classes from seeds. Updates display_name/description/engine_family/sort_order."""
    for seed in CAR_CLASS_SEEDS:
        stmt = pg_insert(CarClass).values(**seed)
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
    await session.commit()


async def init() -> None:
    await create_tables()
    async with _session_factory() as session:
        await seed_car_classes(session)


if __name__ == "__main__":
    asyncio.run(init())
