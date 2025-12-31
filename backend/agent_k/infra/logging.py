"""Centralized logging utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Third-party (alphabetical)
import logfire

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = ("get_logger",)


# =============================================================================
# Section 12: Functions
# =============================================================================
def get_logger(component: str) -> logfire.Logfire:
    """Return a component-specific logger."""
    return logfire.with_settings(tags=[f"component:{component}"])
