"""Centralized logging utilities.

@notice: |
    Centralized logging utilities.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.infra.logging
    provides:
        - agent_k.infra.logging
    pattern: logging

@agent-guidance:
    do:
        - "Use agent_k.infra.logging as the canonical home for this capability."
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

import logfire

__all__ = ("get_logger",)


def get_logger(component: str) -> logfire.Logfire:
    """Return a component-specific logger."""
    return logfire.with_settings(tags=[f"component:{component}"])
