"""Command-line entry point for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# Local imports (core first, then alphabetical)
from . import __version__

__all__ = ("main",)


def main() -> None:
    """Simple CLI entry that reports the package version."""
    print(f"AGENT-K version {__version__}")


if __name__ == "__main__":
    main()
