"""AGENT-K package initialization.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Local imports (core first, then alphabetical)
from ._version import __version__
from .agents.evolver import evolver_agent
from .agents.lobbyist import lobbyist_agent
from .agents.lycurgus import LycurgusOrchestrator
from .agents.scientist import scientist_agent

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "__version__",
    "lobbyist_agent",
    "scientist_agent",
    "evolver_agent",
    "LycurgusOrchestrator",
)
