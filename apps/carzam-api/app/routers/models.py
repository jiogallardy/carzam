"""Lists model checkpoints available in R2 so the mobile app can pick one."""
from fastapi import APIRouter, Depends

from app import inference
from app.models import User
from app.schemas import ModelInfo, ModelsResponse
from app.security import current_user

router = APIRouter(tags=["models"])


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
