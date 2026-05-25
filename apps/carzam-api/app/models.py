"""Database tables. See spec section 2 — keep this file as the single source of truth."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    apple_sub: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owned_cars: Mapped[list["OwnedCar"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    clips: Mapped[list["Clip"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    discoveries: Mapped[list["Discovery"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class CarClass(Base):
    __tablename__ = "car_classes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # 'ferrari_812'
    display_name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    engine_family: Mapped[str | None] = mapped_column(String(120), nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)


class OwnedCar(Base):
    __tablename__ = "owned_cars"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    year: Mapped[int] = mapped_column(Integer)
    make: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(64))
    trim: Mapped[str | None] = mapped_column(String(64), nullable=True)
    color: Mapped[str | None] = mapped_column(String(48), nullable=True)
    mods_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="owned_cars")
    clips: Mapped[list["Clip"]] = relationship(back_populates="owned_car")


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    r2_key: Mapped[str] = mapped_column(Text)
    duration_sec: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))  # 'my_car' | 'recognize' | 'new_car_submission'

    owned_car_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("owned_cars.id", ondelete="SET NULL"), nullable=True
    )
    predicted_class_id: Mapped[str | None] = mapped_column(
        ForeignKey("car_classes.id", ondelete="SET NULL"), nullable=True
    )
    predicted_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    all_probs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    review_status: Mapped[str] = mapped_column(String(16), default="pending")
    user_correction_class_id: Mapped[str | None] = mapped_column(
        ForeignKey("car_classes.id", ondelete="SET NULL"), nullable=True
    )

    submitted_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submitted_make: Mapped[str | None] = mapped_column(String(64), nullable=True)
    submitted_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    submitted_trim: Mapped[str | None] = mapped_column(String(64), nullable=True)
    submitted_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="clips")
    owned_car: Mapped[OwnedCar | None] = relationship(back_populates="clips")
    predicted_class: Mapped[CarClass | None] = relationship(foreign_keys=[predicted_class_id])


class Discovery(Base):
    __tablename__ = "discoveries"
    __table_args__ = (UniqueConstraint("user_id", "car_class_id", name="uq_user_class"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    car_class_id: Mapped[str] = mapped_column(
        ForeignKey("car_classes.id", ondelete="CASCADE"), primary_key=True
    )
    first_clip_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("clips.id", ondelete="SET NULL"), nullable=True
    )
    first_discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    total_matches: Mapped[int] = mapped_column(Integer, default=1)
    best_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    user: Mapped[User] = relationship(back_populates="discoveries")
    car_class: Mapped[CarClass] = relationship()
