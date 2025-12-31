"""AGENT-K package initialization.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

# Local imports (core first, then alphabetical)
from ._version import __version__

if TYPE_CHECKING:
    from .agents.evolver import evolver_agent as evolver_agent
    from .agents.lobbyist import lobbyist_agent as lobbyist_agent
    from .agents.lycurgus import LycurgusOrchestrator as LycurgusOrchestrator
    from .agents.scientist import scientist_agent as scientist_agent

__all__ = ('__version__', 'lobbyist_agent', 'scientist_agent', 'evolver_agent', 'LycurgusOrchestrator')


def __getattr__(name: str) -> Any:
    """Lazy import agents to avoid requiring API keys at import time."""
    if name == 'evolver_agent':
        from .agents.evolver import evolver_agent

        return evolver_agent
    if name == 'lobbyist_agent':
        from .agents.lobbyist import lobbyist_agent

        return lobbyist_agent
    if name == 'scientist_agent':
        from .agents.scientist import scientist_agent

        return scientist_agent
    if name == 'LycurgusOrchestrator':
        from .agents.lycurgus import LycurgusOrchestrator

        return LycurgusOrchestrator
    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
