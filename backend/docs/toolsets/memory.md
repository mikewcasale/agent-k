# Memory Tool Helpers

Agent-K uses pydantic-ai's built-in `MemoryTool` (Anthropic only) with a file-backed memory backend. This avoids reimplementing memory as a custom toolset while still providing persistence.

## Setup

```python
from pydantic_ai import Agent

from agent_k.toolsets import create_memory_backend, prepare_memory_tool, register_memory_tool

memory_backend = create_memory_backend()

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    builtin_tools=[prepare_memory_tool],
)

register_memory_tool(agent, memory_backend)
```

## Usage Notes

- The built-in tool name is `memory`.
- The backend stores files under a base directory (default: `.agent_k_memory`).
- Use create/view commands to persist and retrieve shared context.

Example prompt guidance:

```text
Use the memory tool to create shared/target_competition.md with the chosen competition summary.
```

If the provider does not support `MemoryTool`, `prepare_memory_tool` omits it and the `memory` tool is not registered.
