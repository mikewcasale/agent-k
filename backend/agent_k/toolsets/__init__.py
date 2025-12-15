"""Toolsets for AGENT-K agents.

These toolsets wrap external services (Kaggle, web search, memory)
as pydantic-ai FunctionToolsets that work with ANY model provider.

Unlike builtin tools (WebSearchTool, MCPServerTool) which are executed
by the model provider's infrastructure, these toolsets are executed
client-side by pydantic-ai and thus work with OpenAI-compatible
endpoints like Devstral.
"""
from __future__ import annotations

from .kaggle import create_kaggle_toolset
from .memory import create_memory_toolset
from .search import create_search_toolset

__all__ = [
    'create_kaggle_toolset',
    'create_memory_toolset',
    'create_search_toolset',
]
