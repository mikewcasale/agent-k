"""Centralized logging utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Third-party (alphabetical)
import logfire

__all__ = ("get_logger",)


def get_logger(component: str) -> logfire.Logfire:
    """Return a component-specific logger."""
    return logfire.with_settings(tags=[f"component:{component}"])
