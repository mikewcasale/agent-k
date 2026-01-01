"""Configuration management for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import os
from dataclasses import dataclass

__all__ = ("Settings", "load_settings")


@dataclass
class Settings:
    """Runtime configuration settings."""

    environment: str = os.getenv("ENVIRONMENT", "development")


def load_settings() -> Settings:
    """Load settings from environment."""
    return Settings()
