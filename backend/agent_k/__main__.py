"""Command-line entry point for AGENT-K."""
from __future__ import annotations

from . import __version__

__all__ = ['main']


def main() -> None:
    """Simple CLI entry that reports the package version."""
    print(f'AGENT-K version {__version__}')


if __name__ == '__main__':
    main()
