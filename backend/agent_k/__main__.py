"""Command-line entry point for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Local imports (core first, then alphabetical)
from . import __version__

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = ("main",)


# =============================================================================
# Section 12: Functions
# =============================================================================
def main() -> None:
    """Simple CLI entry that reports the package version."""
    print(f"AGENT-K version {__version__}")


if __name__ == "__main__":
    main()
