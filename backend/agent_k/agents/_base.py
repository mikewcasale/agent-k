"""Base agent patterns for AGENT-K.

Provides reusable patterns and mixins for agent implementations.
All agents use builtin tools for standard capabilities per spec Section 7.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic

import httpx
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from ..core.types import AgentDepsT, OutputT

__all__ = ['BaseAgentMixin', 'AgentDeps']


@dataclass
class AgentDeps:
    """Base dependency container for all agents.
    
    Per spec Section 7.1, dependencies are injected via dataclass containers.
    """
    
    http_client: httpx.AsyncClient
    event_emitter: Any = None  # Will be EventEmitter type
    memory_store: dict[str, Any] = field(default_factory=dict)


class BaseAgentMixin(ABC, Generic[AgentDepsT, OutputT]):
    """Base mixin providing common agent functionality.
    
    Per spec Section 3.2, class structure follows visibility-based ordering.
    """
    
    # =========================================================================
    # Class Variables
    # =========================================================================
    _default_model: str = 'anthropic:claude-sonnet-4-5'
    
    # =========================================================================
    # Abstract Methods
    # =========================================================================
    @abstractmethod
    def _create_agent(self, model: str) -> Agent[AgentDepsT, OutputT]:
        """Create the underlying Pydantic-AI agent."""
        ...
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    async def run(
        self,
        prompt: str,
        *,
        deps: AgentDepsT,
    ) -> OutputT:
        """Execute the agent with the given prompt.
        
        Per spec, all public methods include comprehensive docstrings.
        
        Args:
            prompt: Natural language instruction for the agent.
            deps: Dependency container with required services.
        
        Returns:
            Agent output of type OutputT.
        """
        ...
    
    # =========================================================================
    # Protected Methods
    # =========================================================================
    def _get_agent_name(self) -> str:
        """Return agent name for logging."""
        return self.__class__.__name__
