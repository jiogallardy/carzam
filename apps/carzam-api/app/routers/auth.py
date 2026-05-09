"""Auth — Google + Apple Sign-In → server-issued JWT."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas import AppleSignInRequest, AuthResponse, GoogleSignInRequest, UserOut
from app.security import (
    get_or_create_user_from_apple,
    get_or_create_user_from_google,
    issue_jwt,
    verify_apple_id_token,
    verify_google_id_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/google", response_model=AuthResponse)
async def google_sign_in(
    req: GoogleSignInRequest,
    session: AsyncSession = Depends(get_session),
) -> AuthResponse:
    payload = verify_google_id_token(req.id_token)
    user = await get_or_create_user_from_google(session, payload)
    await session.commit()
    return AuthResponse(jwt=issue_jwt(user.id), user=UserOut.model_validate(user))


@router.post("/apple", response_model=AuthResponse)
async def apple_sign_in(
    req: AppleSignInRequest,
    session: AsyncSession = Depends(get_session),
) -> AuthResponse:
    payload = await verify_apple_id_token(req.id_token)
    user = await get_or_create_user_from_apple(
        session, payload, full_name=req.full_name, email_override=req.email
    )
    await session.commit()
    return AuthResponse(jwt=issue_jwt(user.id), user=UserOut.model_validate(user))
