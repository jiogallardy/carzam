"""Runtime config. All values come from env (Coolify injects these)."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    database_url: str = Field(..., description="postgresql+asyncpg://...")

    # JWT
    jwt_secret: str = Field(..., min_length=32)
    jwt_alg: str = "HS256"
    jwt_ttl_days: int = 30

    # Google OAuth
    google_client_id_ios: str = Field(...)
    google_client_id_web: str | None = None  # only if you also build web

    # Apple Sign-In
    apple_team_id: str = Field(...)
    apple_bundle_id: str = Field(default="com.carzam.app")
    # Apple key fetched from JWKS at runtime; no key file needed.

    # Cloudflare R2
    r2_account_id: str = Field(...)
    r2_access_key_id: str = Field(...)
    r2_secret_access_key: str = Field(...)
    r2_bucket: str = Field(default="carzam-clips")
    r2_public_endpoint: str | None = None  # https://<account>.r2.cloudflarestorage.com

    # Inference — checkpoint is fetched from R2 on container start so the
    # Docker image stays small and we can swap checkpoints without rebuilding.
    checkpoint_path: Path = Field(default=Path("/data/checkpoint.pt"))
    classes_path: Path = Field(default=Path("/data/classes.json"))
    model_r2_key: str = Field(default="model/checkpoint.pt")
    classes_r2_key: str = Field(default="model/classes.json")
    model_run_id: str = Field(default="20260507_174723")
    # If set, the API uses this as the active default model id (e.g. "v6" or
    # "20260601_120000"). Falls back to the legacy `model/checkpoint.pt` when
    # unset so existing deploys keep working unchanged.
    default_model_id: str | None = None

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Admin token for /admin/* routes. Generate with: openssl rand -hex 32.
    # Anyone with this token can list user clips, accept/reject them, and
    # create new car_classes — set it in Coolify env, never commit.
    admin_token: str | None = None

    @property
    def r2_endpoint(self) -> str:
        return self.r2_public_endpoint or f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


@lru_cache
def settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
