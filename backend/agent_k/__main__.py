"""Command-line entry point for AGENT-K.

@notice: |
    Command-line entry point for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.__main__
    provides:
        - agent_k.__main__
    pattern: cli-entrypoint

@agent-guidance:
    do:
        - "Use agent_k.__main__ as the canonical home for this capability."
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

from . import __version__

__all__ = ("main",)


def main() -> None:
    """Simple CLI entry that reports the package version.

    @notice: Prints the current AGENT-K version to stdout.
    @dev: Entry point for `python -m agent_k` invocation.
    """
    print(f"AGENT-K version {__version__}")


if __name__ == "__main__":
    main()
