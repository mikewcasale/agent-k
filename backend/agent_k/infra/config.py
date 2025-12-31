"""Configuration management for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
import os
from dataclasses import dataclass

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = ("Settings", "load_settings")


# =============================================================================
# Section 9: Dataclasses
# =============================================================================
# =============================================================================
# Section 9: Dataclasses
# =============================================================================
@dataclass
class Settings:
    """Runtime configuration settings."""

    environment: str = os.getenv("ENVIRONMENT", "development")


# =============================================================================
# Section 12: Functions
# =============================================================================
# =============================================================================
# Section 12: Functions
# =============================================================================
def load_settings() -> Settings:
    """Load settings from environment."""
    return Settings()
