"""Base settings configuration.

@notice: |
    Base settings configuration.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.settings
    provides:
        - agent_k.core.settings
    pattern: settings

@agent-guidance:
    do:
        - "Use agent_k.core.settings as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ("AgentKSettings", "SCHEMA_VERSION")

SCHEMA_VERSION: Final[str] = "1.0.0"


class AgentKSettings(BaseSettings):
    """Base settings with shared environment defaults.

    @notice: |
        Base settings with shared environment defaults.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: settings-model
            rationale: "Centralizes shared environment defaults."
            violations: "Scattered env lookups drift across modules."
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", validate_default=True)
