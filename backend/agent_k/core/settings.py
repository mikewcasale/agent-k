"""Base settings configuration.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from typing import Final

# Third-party (alphabetical)
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ('AgentKSettings', 'SCHEMA_VERSION')

SCHEMA_VERSION: Final[str] = '1.0.0'


class AgentKSettings(BaseSettings):
    """Base settings with shared environment defaults."""

    model_config = SettingsConfigDict(env_file='.env', extra='ignore', validate_default=True)
