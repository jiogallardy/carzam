"""Read-only access to the model's class set."""
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import CarClass
from app.schemas import CarClassOut

router = APIRouter(tags=["classes"])


@router.get("/car-classes", response_model=list[CarClassOut])
async def list_car_classes(session: AsyncSession = Depends(get_session)) -> list[CarClassOut]:
    res = await session.execute(
        select(CarClass).where(CarClass.id != "other").order_by(CarClass.sort_order, CarClass.id)
    )
    return [CarClassOut.model_validate(c) for c in res.scalars().all()]
