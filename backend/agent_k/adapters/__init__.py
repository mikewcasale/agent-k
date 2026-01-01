"""Platform adapter implementations.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Local imports (core first, then alphabetical)
from agent_k.core.protocols import PlatformAdapter

from .kaggle import KaggleAdapter, KaggleSettings
from .openevolve import OpenEvolveAdapter, OpenEvolveSettings

__all__ = (
    "PlatformAdapter",
    "KaggleAdapter",
    "KaggleSettings",
    "OpenEvolveAdapter",
    "OpenEvolveSettings",
)
