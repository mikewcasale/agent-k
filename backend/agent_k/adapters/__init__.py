"""Platform adapter implementations.

@notice: |
    Platform adapter implementations.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.adapters
    provides:
        - agent_k.adapters
    pattern: adapter-package

@agent-guidance:
    do:
        - "Use agent_k.adapters as the canonical home for this capability."
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

from agent_k.core.protocols import PlatformAdapter

from .kaggle import KaggleAdapter, KaggleSettings
from .openevolve import OpenEvolveAdapter, OpenEvolveRunner, OpenEvolveSettings

__all__ = (
    "PlatformAdapter",
    "KaggleAdapter",
    "KaggleSettings",
    "OpenEvolveAdapter",
    "OpenEvolveRunner",
    "OpenEvolveSettings",
)
