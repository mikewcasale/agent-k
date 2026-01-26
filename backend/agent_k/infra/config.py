"""Configuration management for AGENT-K.

@notice: |
    Configuration management for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.infra.config
    provides:
        - agent_k.infra.config
    pattern: config

@agent-guidance:
    do:
        - "Use agent_k.infra.config as the canonical home for this capability."
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

import os
from dataclasses import dataclass

__all__ = ("Settings", "load_settings")


@dataclass
class Settings:
    """Runtime configuration settings.

    @pattern:
        name: settings-model
        rationale: "Captures runtime configuration defaults."
        violations: "Scattered env access increases drift."
    """

    environment: str = os.getenv("ENVIRONMENT", "development")


def load_settings() -> Settings:
    """Load settings from environment.

    @notice: |
        Creates a Settings instance from environment variables.

    @dev: |
        Currently only reads ENVIRONMENT variable. Extend as needed.
    """
    return Settings()
