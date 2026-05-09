"""FastAPI entry point."""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import inference, init_db
from app.config import settings
from app.routers import admin, auth, classes, me, models, recognize

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("init_db")
    await init_db.init()
    log.info("loading_model", path=str(settings().checkpoint_path))
    inference.load_model()
    cars, _states = inference.classes()
    log.info("model_loaded", run_id=inference.model_run_id(), n_cars=len(cars))
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Carzam API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings().cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {
            "status": "ok",
            "model_run_id": inference.model_run_id(),
            "default_model_id": inference.default_id(),
        }

    app.include_router(auth.router)
    app.include_router(me.router)
    app.include_router(recognize.router)
    app.include_router(classes.router)
    app.include_router(models.router)
    app.include_router(admin.router)
    return app


app = create_app()
