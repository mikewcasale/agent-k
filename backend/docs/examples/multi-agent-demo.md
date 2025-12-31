# Multi-Agent Demo

This example demonstrates AGENT-K's multi-agent system with the LOBBYIST (discovery) and SCIENTIST (research) agents working together.

## Overview

The `multi_agent_playbook.py` script runs through two phases:

1. **Discovery**: LOBBYIST searches for Kaggle competitions
2. **Research**: SCIENTIST analyzes the selected competition

## Running the Demo

```bash
cd backend

# With Claude Haiku (Anthropic)
uv run python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307

# With GPT-4o (OpenAI)
uv run python examples/multi_agent_playbook.py --model openai:gpt-4o
```

## Key Built-in Tools

The demo uses built-in tools when supported by the provider:

- `WebSearchTool` for web search (`web_search`)
- `MemoryTool` for cross-agent memory (Anthropic only)

Kaggle access is still provided via the `kaggle_toolset` FunctionToolset.

## Memory Backend (Anthropic Only)

The demo registers a file-backed memory backend:

```python
from agent_k.toolsets import create_memory_backend, prepare_memory_tool, register_memory_tool

memory_backend = create_memory_backend()
agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    builtin_tools=[prepare_memory_tool],
)
register_memory_tool(agent, memory_backend)
```

## Prompts

The prompts instruct agents to use the built-in `web_search` tool and (when available) the `memory` tool for sharing notes.

For the full code, see `backend/examples/multi_agent_playbook.py`.
