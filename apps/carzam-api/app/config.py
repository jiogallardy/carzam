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

    # Inference
    checkpoint_path: Path = Field(default=Path("/data/checkpoint.pt"))
    classes_path: Path = Field(default=Path("/data/classes.json"))

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def r2_endpoint(self) -> str:
        return self.r2_public_endpoint or f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


@lru_cache
def settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
