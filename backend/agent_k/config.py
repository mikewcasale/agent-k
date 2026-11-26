"""Runtime settings, loaded explicitly by API and worker entry points.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Final

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ("MAX_MISSION_EVOLUTION_ROUNDS", "Settings", "load_settings")

MAX_MISSION_EVOLUTION_ROUNDS: Final[int] = 200


class Settings(BaseSettings):
    """Validated configuration for a mission process.

    Environment names use the ``AGENT_K_`` prefix. Provider and Kaggle credentials
    stay in the worker environment; none are forwarded to candidate containers.
    """

    model_config = SettingsConfigDict(env_prefix="AGENT_K_", extra="ignore")

    environment: str = "development"
    model: str = "devstral:local"
    database_url: SecretStr | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    gateway_secret: SecretStr | None = Field(default=None, min_length=32)
    gateway_previous_secret: SecretStr | None = Field(default=None, min_length=32)
    gateway_issuer: str = "agent-k-frontend"
    gateway_audience: str = "agent-k-api"
    gateway_token_ttl_seconds: int = Field(default=60, ge=30, le=300)
    worker_token: SecretStr | None = Field(default=None, min_length=32)
    worker_previous_token: SecretStr | None = Field(default=None, min_length=32)
    allow_submissions: bool = False
    phase_timeout_seconds: int = Field(default=3600, ge=60)
    evolution_iterations: int = Field(default=20, ge=1, le=MAX_MISSION_EVOLUTION_ROUNDS)
    evaluation_timeout_seconds: int = Field(default=1200, ge=1)
    evaluation_parallelism: int = Field(default=1, ge=1, le=16)
    sandbox_image: str | None = None
    gpu_devices: tuple[str, ...] = ()
    kumo_cache: Path | None = None
    vision_weights: Path | None = None
    transformer_cache: Path | None = None
    memory_root: Path = Path("/tmp/agent-k/memory")


def load_settings() -> Settings:
    """Read current environment values without an import-time singleton."""
    return Settings()
