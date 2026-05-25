"""Authenticated user routes: /me, /me/cars, /me/discoveries, stats."""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_session
from app.models import CarClass, Clip, Discovery, OwnedCar, User
from app.schemas import (
    CarClassOut,
    DiscoveryOut,
    OwnedCarCreate,
    OwnedCarOut,
    StatsOut,
    UserOut,
)
from app.security import current_user

router = APIRouter(tags=["me"])


@router.get("/me", response_model=UserOut)
async def get_me(user: User = Depends(current_user)) -> UserOut:
    return UserOut.model_validate(user)


@router.get("/me/cars", response_model=list[OwnedCarOut])
async def list_cars(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> list[OwnedCarOut]:
    res = await session.execute(
        select(OwnedCar).where(OwnedCar.user_id == user.id).order_by(OwnedCar.created_at.desc())
    )
    return [OwnedCarOut.model_validate(c) for c in res.scalars().all()]


@router.post("/me/cars", response_model=OwnedCarOut, status_code=201)
async def add_car(
    body: OwnedCarCreate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> OwnedCarOut:
    car = OwnedCar(user_id=user.id, **body.model_dump())
    session.add(car)
    await session.commit()
    await session.refresh(car)
    return OwnedCarOut.model_validate(car)


@router.delete("/me/cars/{car_id}", status_code=204)
async def delete_car(
    car_id: uuid.UUID,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    car = await session.get(OwnedCar, car_id)
    if car is None or car.user_id != user.id:
        raise HTTPException(404, "car not found")
    await session.delete(car)
    await session.commit()


@router.get("/me/discoveries", response_model=list[DiscoveryOut])
async def list_discoveries(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> list[DiscoveryOut]:
    res = await session.execute(
        select(Discovery)
        .where(Discovery.user_id == user.id)
        .options(selectinload(Discovery.car_class))
        .order_by(Discovery.first_discovered_at.desc())
    )
    return [
        DiscoveryOut(
            car_class=CarClassOut.model_validate(d.car_class),
            first_discovered_at=d.first_discovered_at,
            total_matches=d.total_matches,
            best_confidence=d.best_confidence,
        )
        for d in res.scalars().all()
    ]


@router.get("/me/stats", response_model=StatsOut)
async def stats(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> StatsOut:
    n_total = (await session.execute(
        select(func.count(CarClass.id)).where(CarClass.id != "other")
    )).scalar_one()
    n_discovered = (await session.execute(
        select(func.count(Discovery.car_class_id))
        .where(Discovery.user_id == user.id, Discovery.car_class_id != "other")
    )).scalar_one()
    total_recogs = (await session.execute(
        select(func.count(Clip.id)).where(Clip.user_id == user.id, Clip.source == "recognize")
    )).scalar_one()
    pct = (100.0 * n_discovered / n_total) if n_total else 0.0
    return StatsOut(
        n_discovered=int(n_discovered),
        n_total=int(n_total),
        completion_percent=round(pct, 1),
        total_recognitions=int(total_recogs),
    )
