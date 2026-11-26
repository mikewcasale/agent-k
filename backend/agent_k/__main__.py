"""Command-line entry points for AGENT-K.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse

from .version import __version__

__all__ = ("main",)


def main(argv: list[str] | None = None) -> None:
    """Report the version or dispatch an explicit runtime command."""
    parser = argparse.ArgumentParser(prog="agent-k")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("command", nargs="?", choices=("worker", "migrate", "provision-kumo"))
    arguments, remainder = parser.parse_known_args(argv)
    if arguments.version or arguments.command is None:
        print(f"AGENT-K version {__version__}")
        return
    if arguments.command == "worker":
        from .worker import main as run_worker

        raise SystemExit(run_worker(remainder))
    elif arguments.command == "migrate":
        from .migration import main as run_migration

        raise SystemExit(run_migration(remainder))
    else:
        from .tabular import main as run_tabular

        remainder.insert(0, "provision-kumo")
        raise SystemExit(run_tabular(remainder))


if __name__ == "__main__":
    main()
