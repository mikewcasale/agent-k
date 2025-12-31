"""Base settings configuration.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
from typing import Final

# Third-party (alphabetical)
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = ("AgentKSettings", "SCHEMA_VERSION")

# =============================================================================
# Section 3: Constants
# =============================================================================
SCHEMA_VERSION: Final[str] = "1.0.0"


# =============================================================================
# Section 11: Classes
# =============================================================================
# =============================================================================
# Section 11: Classes
# =============================================================================
class AgentKSettings(BaseSettings):
    """Base settings with shared environment defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )
