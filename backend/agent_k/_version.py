"""Version information for the AGENT-K package.

@notice: |
    Internal version module.

@dev: |
    Use `from agent_k import __version__` instead of importing directly.

@graph:
    id: agent_k._version
    provides:
        - agent_k._version:__version__
    pattern: versioning

@agent-guidance:
    do:
        - "Use agent_k._version as the canonical version source."
    do_not:
        - "Duplicate version constants elsewhere."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Final

__all__ = ("__version__",)

__version__: Final[str] = "0.1.0"
