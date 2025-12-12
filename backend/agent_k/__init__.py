"""AGENT-K package initialization.

This package implements the AGENT-K multi-agent Kaggle competition framework
as defined by the python_spec_v2 specification.
"""
from __future__ import annotations

from ._version import __version__
from .agents.evolver.agent import EvolverAgent
from .agents.lobbyist.agent import LobbyistAgent
from .agents.lycurgus.agent import LycurgusOrchestrator
from .agents.scientist.agent import ScientistAgent

__all__ = [
    '__version__',
    'LobbyistAgent',
    'ScientistAgent',
    'EvolverAgent',
    'LycurgusOrchestrator',
]
