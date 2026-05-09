"""JWT issuance + verification, plus Google/Apple ID-token verification."""
from __future__ import annotations

import time
import uuid

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.models import User

bearer = HTTPBearer(auto_error=True)


# ===================== Carzam JWT =====================

def issue_jwt(user_id: uuid.UUID) -> str:
    cfg = settings()
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + cfg.jwt_ttl_days * 86400,
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_alg)


def decode_jwt(token: str) -> uuid.UUID:
    cfg = settings()
    try:
        payload = jwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_alg])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from e
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    return uuid.UUID(sub)


async def current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    session: AsyncSession = Depends(get_session),
) -> User:
    user_id = decode_jwt(creds.credentials)
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")
    return user


# ===================== Google ID token =====================

def verify_google_id_token(id_token_str: str) -> dict:
    """Returns the validated payload (incl. 'sub', 'email', 'name')."""
    cfg = settings()
    try:
        payload = google_id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            audience=cfg.google_client_id_ios,
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"google token invalid: {e}") from e
    if payload.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "google issuer mismatch")
    return payload


# ===================== Apple ID token =====================

_APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
_apple_jwks_cache: dict | None = None
_apple_jwks_fetched_at: float = 0.0


async def _fetch_apple_jwks() -> dict:
    global _apple_jwks_cache, _apple_jwks_fetched_at
    if _apple_jwks_cache and (time.time() - _apple_jwks_fetched_at) < 3600:
        return _apple_jwks_cache
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(_APPLE_JWKS_URL)
        r.raise_for_status()
    _apple_jwks_cache = r.json()
    _apple_jwks_fetched_at = time.time()
    return _apple_jwks_cache


async def verify_apple_id_token(id_token_str: str) -> dict:
    """Returns Apple's claims dict (incl. 'sub', 'email')."""
    cfg = settings()
    headers = jwt.get_unverified_headers(id_token_str)
    kid = headers.get("kid")
    if not kid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "apple token missing kid")

    jwks = await _fetch_apple_jwks()
    key = next((k for k in jwks.get("keys", []) if k["kid"] == kid), None)
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "apple key not found for kid")

    try:
        payload = jwt.decode(
            id_token_str,
            key,
            algorithms=[key.get("alg", "RS256")],
            audience=cfg.apple_bundle_id,
            issuer="https://appleid.apple.com",
        )
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"apple token invalid: {e}") from e
    return payload


async def get_or_create_user_from_google(session: AsyncSession, payload: dict) -> User:
    sub = payload["sub"]
    email = payload.get("email")
    name = payload.get("name")
    res = await session.execute(select(User).where(User.google_sub == sub))
    user = res.scalar_one_or_none()
    if user is None:
        user = User(google_sub=sub, email=email, display_name=name)
        session.add(user)
        await session.flush()
    return user


async def get_or_create_user_from_apple(
    session: AsyncSession,
    payload: dict,
    full_name: str | None,
    email_override: str | None,
) -> User:
    sub = payload["sub"]
    # Apple only provides email/name on first sign-in. Fall back to whatever client sent.
    email = payload.get("email") or email_override
    res = await session.execute(select(User).where(User.apple_sub == sub))
    user = res.scalar_one_or_none()
    if user is None:
        user = User(apple_sub=sub, email=email, display_name=full_name)
        session.add(user)
        await session.flush()
    else:
        # First-time-only data may have arrived now — backfill.
        if full_name and not user.display_name:
            user.display_name = full_name
        if email and not user.email:
            user.email = email
    return user
