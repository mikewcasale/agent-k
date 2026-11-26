# Python AI Framework Style Guide (Compressed)

> Coding conventions for Python AI applications using Pydantic AI, Pydantic Graph, Pydantic Evals, and Logfire. Project-specific guides take precedence. **Consistency over correctness.**

---

## 1. Code Layout

Formatting standards for indentation, spacing, imports, and structure.

### Rules

- Use **4 spaces** per indentation level (never tabs)
- **Two blank lines** between top-level declarations; **one blank line** between methods within a class
- Blank lines may be omitted between groups of related one-liners (e.g., abstract method stubs, enum members)
- **Line length**: 88-100 characters max (Black/Ruff default: 88)
- Wrapped lines: no args attached to opening paren, single indent, closing paren on own line
- **UTF-8 encoding** (Python 3 default, no declaration needed)
- **Trailing commas** required in all multi-line structures
- **Double quotes** for strings (consistent with JSON)
- **f-strings** over `.format()` or `%` formatting

### Import Order

1. `from __future__ import annotations as _annotations` (always first, aliased)
2. Standard library imports (alphabetized)
3. Third-party imports (alphabetized)
4. Local application imports (relative preferred)
5. `TYPE_CHECKING` imports (circular dependency avoidance)

```python
from __future__ import annotations as _annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import Self, TypeAliasType

from .._output import OutputObjectDefinition
from ..exceptions import UserError

if TYPE_CHECKING:
    from ..models.instrumented import InstrumentationSettings
```

### Optional Dependency Handling

```python
try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
```

### Function/Method Ordering

1. Public methods (external API)
2. Protected methods (`_method`)
3. Private methods (`__method`, use sparingly)

Within each group: class methods → properties → instance methods

### Whitespace Rules

- No extraneous whitespace inside `()`, `[]`, `{}`
- No space before `,`, `;`, `:`
- Surround operators with single space: `x = 1`, `x = x * 2 - 1`

### Control Structures

- Use early return pattern over deep nesting
- Single space after `if`, `elif`, `else`, `while`, `for`, `try`, `except`, `with`

```python
# Yes - Early return
async def validate_request(self, request: Request) -> bool:
    if not request.headers.get("Authorization"):
        return False
    if not await self.check_permission(request):
        return False
    return True
```

### Function Declarations

- Short functions: body on same line
- Long functions: use keyword-only args with `*`
- Decorator order: `@classmethod/@staticmethod` → `@property` → `@abstractmethod` → `@override` → `@deprecated` → custom

```python
def __init__(
    self,
    *,
    name: str | None = None,
    state_type: type[StateT] = type(None),
    auto_instrument: bool = True,
) -> None:
    ...
```

---

## 2. Type Hints

Modern Python 3.11+ type annotation standards.

### Rules

- Use **lowercase built-in generics**: `list[str]`, `dict[str, Any]`, `set[int]`, `tuple[int, ...]`
- Import from `collections.abc`: `Callable`, `Iterator`, `Sequence`, `Mapping`, `AsyncIterator`
- Use **union operator** `|` instead of `Union`/`Optional`: `str | int | None`
- Use `TypeAliasType` for complex type aliases
- Use `NewType` for semantic type wrappers: `AgentId = NewType("AgentId", str)`

### Keep from `typing`

```python
from typing import (
    TYPE_CHECKING,  # Conditional imports
    Any,            # No lowercase equivalent
    Generic,        # Generic classes
    Literal,        # Literal types
    TypeVar,        # Type variables
    overload,       # Function overloads
    cast,           # Type casting
)
```

### TypeVar Conventions

```python
StateT = TypeVar("StateT", infer_variance=True)
"""Type variable for graph state."""

DepsT = TypeVar("DepsT", infer_variance=True)
"""Type variable for graph dependencies."""

# Covariant (return types)
RunEndT = TypeVar("RunEndT", covariant=True, default=None)

# Contravariant (input types)
InputsT = TypeVar("InputsT", default=Any, contravariant=True)
```

---

## 3. Project Structure

File organization and module layout standards.

### Rules

- Prefer **flat directory structure** with minimal nesting
- Combine related components into single files
- File names: **single lowercase words without underscores** (`agents.py`, not `agent_manager.py`)
- Split files only when: >3500 lines, distinct domains, circular imports, or base classes needed by multiple implementations
- Abstract base classes/interfaces go in dedicated `base.py` or `interfaces.py` files

### Recommended Structure

```
myproject/
├── src/mypackage/
│   ├── __init__.py
│   ├── agents.py      # Agent classes, runners, related functions
│   ├── models.py      # Pydantic models, dataclasses, TypedDicts
│   ├── types.py       # TypeVars, TypeAliases, NewTypes, Protocols
│   ├── stores.py      # All store implementations
│   ├── base.py        # Abstract base classes ONLY
│   ├── config.py      # Settings, constants, configuration
│   └── exceptions.py  # All exception classes
├── tests/
└── pyproject.toml
```

### Module Layout Order

| Order | Element | Required |
|-------|---------|----------|
| 1 | Module docstring with copyright | **YES** |
| 2 | `from __future__ import annotations as _annotations` | **YES** |
| 3 | Standard library imports | YES |
| 4 | Third-party imports | YES |
| 5 | Local imports | YES |
| 6 | `TYPE_CHECKING` imports | If needed |
| 7 | Module-level TypeVars (with docstrings) | If needed |
| 8 | `__all__` tuple | **YES** |
| 9 | Module-level constants (UPPER_CASE) | If needed |
| 10 | Enums | If needed |
| 11 | Protocols/ABCs | If needed |
| 12 | Dataclasses/Models | If needed |
| 13 | Concrete classes | YES |
| 14 | Module-level functions | If needed |

### Class Layout Order

| Order | Element |
|-------|---------|
| 1 | Class docstring |
| 2 | Class-level constants |
| 3 | Public instance attributes (with docstrings) |
| 4 | Protected instance attributes (`_prefix`) |
| 5 | Class-level TypeAliases |
| 6 | `__init__` |
| 7 | `__aenter__`/`__aexit__` |
| 8 | `__enter__`/`__exit__` |
| 9 | Properties |
| 10 | `@overload` signatures |
| 11 | Class methods |
| 12 | Static methods |
| 13 | Public methods |
| 14 | Protected methods |
| 15 | Magic methods (`__repr__`, etc.) |

---

## 4. Naming Conventions

Naming standards for identifiers throughout the codebase.

### Style Reference

| Style | Usage |
|-------|-------|
| `lowercase` | Module/file names |
| `lower_case_with_underscores` | Functions, variables, methods |
| `UPPER_CASE_WITH_UNDERSCORES` | Module constants |
| `CapitalizedWords` | Classes, TypeVars, type aliases |
| `_single_leading_underscore` | Protected (internal) |
| `single_trailing_underscore_` | Keyword conflict avoidance (`type_`, `id_`) |

### Rules

- **Avoid**: single chars unless complex math formula (except `i`, `j`, `k`, `T`, `K`, `V`), shadowing built-ins, ambiguous abbreviations, generic names without context
- **Initialisms**: capitalize all letters (`HTTPServerError`, not `HttpServerError`)
- **Exceptions**: end with `Error` or `Exception`
- **Protocols/ABCs**: descriptive names (`AbstractToolset`, `AgentStore`, `BaseStatePersistence`)
- **TypeVars**: short names ending with `T` (`StateT`, `DepsT`, `InputT`)
- **Enums**: CapWords class, string values for serialization

### Visibility Guidelines

| Visibility | Prefix | Usage |
|------------|--------|-------|
| **Public** | None | Stable API, documented, tested |
| **Protected** | `_single` | Internal, may be used by subclasses |
| **Private** | `__double` | Name mangling, use sparingly |

---

## 5. Docstrings

Documentation standards using Google style format.

### Rules

- All public interfaces must have docstrings
- Use triple double quotes (`"""..."""`)
- Module docstrings: brief description + copyright + license + optional author
- Class docstrings: description + type parameters + example
- Method docstrings: description + Args + Returns + Raises
- Attribute docstrings: inline after declaration

### Module Docstring Template

```python
"""Brief description of the module.

Extended description with more details.

(c) MyCompany 2025.
All rights reserved.

Author: Jane Developer <jane@mycompany.com>
"""
```

### Class Docstring Template

```python
class GraphBuilder(Generic[StateT, DepsT]):
    """Brief description.

    Extended description.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies

    Example:
        ```python
        builder = GraphBuilder(state_type=MyState)
        graph = builder.build()
        ```
    """
```

### Method Docstring Template

```python
async def request(self, messages: list[ModelMessage]) -> ModelResponse:
    """Brief description.

    Args:
        messages: The conversation history.

    Returns:
        The model's response.

    Raises:
        ModelError: If the request fails.
    """
```

### Attribute Docstrings

```python
@dataclass
class GraphBuilder:
    name: str | None
    """Optional name for the graph."""

    state_type: type[StateT]
    """The type of the graph state."""
```

---

## 6. Pydantic AI Framework

Agent construction, tools, dependency injection, streaming, and testing patterns.

### Rules

- **Prefer built-in tools** (`WebSearchTool`, `CodeExecutionTool`, `MCPServerTool`) over custom implementations
- **Use `RunContext[DepsT]`** for dependency injection into tools and instructions
- **Use `@agent.tool`** for tools needing deps; **`@agent.tool_plain`** for stateless tools
- **Use `output_type`** with Pydantic models for structured output (auto-validated + retried)
- **Use `ModelRetry`** in tools/validators to prompt the LLM to self-correct
- **Use `agent.override()`** in tests to swap models, deps, and toolsets
- **Use async context managers** for resource cleanup
- **Use logfire** for observability (not `logging.getLogger()`)
- **Use pydantic-graph** for state machines with built-in persistence
- **Use pydantic-evals** for agent evaluation

### Agent Construction

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Minimal agent
agent = Agent("openai:gpt-4o")

# Full agent with deps, output type, and instructions
@dataclass
class SupportDeps:
    customer_id: int
    db: DatabaseConn

class SupportOutput(BaseModel):
    advice: str = Field(description="Advice returned to customer")
    block_card: bool
    risk: int = Field(ge=0, le=10)

support_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=SupportDeps,
    output_type=SupportOutput,
    instructions="You are a bank support agent.",
    retries=3,
)
```

### Model Strings

```python
# Format: "provider:model-name"
"openai:gpt-4o"
"openai:gpt-5.2"
"anthropic:claude-sonnet-4-6"
"google-gla:gemini-2.0-flash"
"deepseek:deepseek-chat"

# Via Pydantic AI Gateway
"gateway/openai:gpt-5.2"
"gateway/anthropic:claude-sonnet-4-6"

# Override model at runtime
result = agent.run_sync("hello", model="openai:gpt-4o")
```

### Running Agents

```python
# Synchronous
result = agent.run_sync("What is 2+2?")
print(result.output)

# Asynchronous
result = await agent.run("What is 2+2?", deps=my_deps)
print(result.output)

# Streaming text
async with agent.run_stream("Tell me a story") as response:
    async for text in response.stream_text():
        print(text, end="")

# Streaming structured output
async with agent.run_stream("Create a record") as response:
    async for partial in response.stream_output():
        print(partial)  # Incrementally validated

# Graph iteration (full control over agent nodes)
async with agent.iter("What is the weather?") as run:
    async for node in run:
        if Agent.is_model_request_node(node):
            async with node.stream(run.ctx) as stream:
                async for event in stream:
                    print(event)
        elif Agent.is_end_node(node):
            print(run.result.output)
```

### Instructions (Static & Dynamic)

```python
# Static — keyword arg
agent = Agent("openai:gpt-4o", instructions="Be concise.")

# Dynamic — decorator with deps access
@agent.instructions
def add_context(ctx: RunContext[SupportDeps]) -> str:
    return f"Customer ID: {ctx.deps.customer_id}"

# Multiple instructions stack (all are sent as system prompts)
@agent.instructions
def add_date() -> str:
    from datetime import date
    return f"Today is {date.today()}."

# Runtime override
result = agent.run_sync("Hi", instructions="Be extra friendly")
```

### Tools

```python
from pydantic_ai import Agent, RunContext, ModelRetry, Tool

agent = Agent("openai:gpt-4o", deps_type=SupportDeps)

# Tool with deps (docstring → LLM tool description)
@agent.tool
async def customer_balance(
    ctx: RunContext[SupportDeps], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )

# Plain tool (no deps)
@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die."""
    return str(random.randint(1, 6))

# Tool with retries + ModelRetry for self-correction
@agent.tool(retries=3)
def get_user(ctx: RunContext[SupportDeps], name: str) -> int:
    """Get user ID by full name."""
    user_id = ctx.deps.db.get_user(name=name)
    if user_id is None:
        raise ModelRetry(f"No user {name!r}, try the full name")
    return user_id

# Tool class (for reuse across agents)
agent = Agent("openai:gpt-4o", tools=[Tool(my_function, retries=2)])
```

### Dependency Injection

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent("openai:gpt-4o", deps_type=MyDeps)

@agent.tool
async def fetch_data(ctx: RunContext[MyDeps], query: str) -> str:
    """Dependencies available via ctx.deps — type-safe."""
    resp = await ctx.deps.http_client.get(
        "https://api.example.com/search",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        params={"q": query},
    )
    return resp.text

# Pass deps at runtime
async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps("key123", client)
        result = await agent.run("Search for AI papers", deps=deps)
```

### Structured Output

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class CityLocation(BaseModel):
    city: str
    country: str

# Single model output (auto-validated, retried on failure)
agent = Agent("openai:gpt-4o", output_type=CityLocation)
result = agent.run_sync("Where were the 2012 Olympics?")
print(result.output)  # city='London' country='United Kingdom'

# Union output (multiple possible types)
agent = Agent("openai:gpt-4o", output_type=[Fruit, Vehicle])

# Scalar output
agent = Agent("openai:gpt-4o", output_type=bool)
```

### Output Validators

```python
from pydantic_ai import Agent, RunContext, ModelRetry

agent = Agent("openai:gpt-4o", deps_type=MyDeps)

@agent.output_validator
async def validate_output(ctx: RunContext[MyDeps], output: str) -> str:
    """Runs after output is parsed; raise ModelRetry to re-prompt."""
    if "forbidden" in output.lower():
        raise ModelRetry("Output contains forbidden content, try again")
    return output
```

### Message History & Conversations

```python
agent = Agent("openai:gpt-4o")

# First exchange
result1 = agent.run_sync("Who was Einstein?")

# Continue conversation
result2 = agent.run_sync(
    "What was his most famous equation?",
    message_history=result1.new_messages(),
)

# Access all messages from a result
all_msgs = result1.all_messages()
```

### Result Object

```python
result = agent.run_sync("What is 2+2?")

result.output            # The parsed output (str, BaseModel, etc.)
result.all_messages()    # Full message history
result.new_messages()    # Messages from this run only (for continuation)
result.usage()           # RunUsage(input_tokens=N, output_tokens=N, requests=N)
result.cost()            # Cost info (if available from provider)
```

### Model Settings

```python
from pydantic_ai import Agent, ModelSettings

# Agent-level defaults
agent = Agent("openai:gpt-4o", model_settings=ModelSettings(temperature=0.5))

# Run-level override
result = agent.run_sync(
    "Be creative",
    model_settings=ModelSettings(temperature=0.9, max_tokens=500),
)
```

### Built-in Tools

```python
from pydantic_ai import Agent, WebSearchTool, CodeExecutionTool, MCPServerTool

# Web search
agent = Agent("openai:gpt-4o", builtin_tools=[WebSearchTool()])

# Web search with config
agent = Agent(
    "openai:gpt-4o",
    builtin_tools=[WebSearchTool(user_location={"city": "London"}, max_results=10)],
)

# Code execution
agent = Agent("openai:gpt-4o", builtin_tools=[CodeExecutionTool()])

# MCP server
agent = Agent(
    "anthropic:claude-sonnet-4-6",
    builtin_tools=[MCPServerTool(id="deepwiki", url="https://mcp.deepwiki.com/mcp")],
)

# Conditional builtin tool via prepare function
async def maybe_search(ctx: RunContext[dict]) -> WebSearchTool | None:
    if not ctx.deps.get("location"):
        return None
    return WebSearchTool(user_location={"city": ctx.deps["location"]})

agent = Agent("openai:gpt-4o", builtin_tools=[maybe_search], deps_type=dict)
```

### Tool Preparation (Conditional/Dynamic Tools)

```python
from pydantic_ai import Agent, RunContext, Tool, ToolDefinition
from dataclasses import replace

# Per-tool prepare function
async def require_access(ctx: RunContext[MyDeps]) -> bool:
    return ctx.deps.has_admin_access  # False disables tool for this run

agent = Agent(
    "openai:gpt-4o",
    tools=[Tool(admin_action, prepare=require_access)],
)

# Agent-wide prepare_tools — modify all tool definitions
async def strict_for_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition],
) -> list[ToolDefinition] | None:
    if ctx.model.system == "openai":
        return [replace(td, strict=True) for td in tool_defs]
    return tool_defs

agent = Agent("openai:gpt-4o", prepare_tools=strict_for_openai)
```

### Testing

```python
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo

# TestModel — deterministic responses, tracks tool calls
result = agent.run_sync("hello", model=TestModel())

# FunctionModel — custom response logic
def mock_response(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("mocked")])

result = agent.run_sync("hello", model=FunctionModel(mock_response))

# Override model/deps/tools in tests via context manager
with agent.override(model=TestModel(), deps=test_deps):
    result = agent.run_sync("test query")
```

### Multi-Agent Delegation

```python
router = Agent("openai:gpt-4o", instructions="Route to the right specialist.")
specialist = Agent("openai:gpt-4o", instructions="You are a domain expert.")

@router.tool
async def delegate(ctx: RunContext[None], query: str) -> str:
    """Hand off to the specialist agent."""
    result = await specialist.run(query)
    return result.output
```

### Retries & Reflection

```python
from pydantic_ai import Agent, ModelRetry

# Agent-level retries (applies to output validation + all tools)
agent = Agent("openai:gpt-4o", retries=3)

# Tool-level retries override agent default
@agent.tool(retries=5)
def flaky_tool(ctx: RunContext[None]) -> str:
    raise ModelRetry("Try again with different parameters")

# Structured output auto-retries on validation failure:
# If model returns {"risk": 15} but Field(le=10), pydantic-ai
# sends the ValidationError back to the LLM to self-correct.
```

### Resource Cleanup

```python
class DatabaseConnection:
    async def __aenter__(self) -> Self:
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self._disconnect()
        return False

@asynccontextmanager
async def managed_session(store: SessionStore, sid: SessionId) -> AsyncIterator[Session]:
    session = await store.acquire(sid)
    try:
        yield session
    finally:
        await store.release(sid)
```

### Pydantic Graph

State machine framework using type hints for complex workflows.

```python
from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

# Nodes: @dataclass classes extending BaseNode[StateT, DepsT, RunEndT]
@dataclass
class DivisibleBy5(BaseNode[None, None, int]):
    value: int

    async def run(self, ctx: GraphRunContext) -> Increment | End[int]:
        if self.value % 5 == 0:
            return End(self.value)
        return Increment(self.value)

@dataclass
class Increment(BaseNode):
    value: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.value + 1)

# Build and run
graph = Graph(nodes=[DivisibleBy5, Increment])
result = graph.run_sync(DivisibleBy5(4))
print(result.output)  # 5
```

**Stateful graph with dependencies:**

```python
@dataclass
class OrderState:
    balance: float = 0.0
    product: str | None = None

@dataclass
class GraphDeps:
    db: DatabaseConn

@dataclass
class InsertCoin(BaseNode[OrderState, GraphDeps, str]):
    amount: float

    async def run(self, ctx: GraphRunContext[OrderState, GraphDeps]) -> SelectProduct:
        ctx.state.balance += self.amount
        return SelectProduct()

# Run with state and deps
state = OrderState()
deps = GraphDeps(db=conn)
result = await graph.run(InsertCoin(1.0), state=state, deps=deps)
```

**Graph execution modes:**

```python
# Async run
result = await graph.run(start_node, state=state)

# Sync run
result = graph.run_sync(start_node, state=state)

# Step-by-step iteration
async with graph.iter(start_node, state=state) as run:
    async for node in run:
        print(node, run.state)

# Result access
result.output    # RunEndT value from End()
result.state     # Final state
```

**Persistence (durable execution):**

```python
from pydantic_graph import FullStatePersistence
from pydantic_graph.persistence.file import FileStatePersistence

# In-memory with history
persistence = FullStatePersistence()
result = await graph.run(node, state=state, persistence=persistence)
for step in persistence.history:
    print(step.node, step.state)

# File-based (survives restarts)
persistence = FileStatePersistence(Path(f"run_{run_id}.json"))
await graph.initialize(start_node, state=state, persistence=persistence)

# Resume from file
async with graph.iter_from_persistence(persistence) as run:
    node_or_end = await run.next()
```

**Edge labels for diagrams:**

```python
from pydantic_graph import Edge
from typing import Annotated

async def run(self, ctx: GraphRunContext[MyState]) -> (
    Annotated[End[str], Edge(label="success")] | Retry
):
    return End("done")

# Generate Mermaid diagram
print(graph.mermaid_code(start_node=MyNode))
```

**Graph + Agent integration:**

```python
from pydantic_ai import Agent
from pydantic_graph import BaseNode, Graph, GraphRunContext

ask_agent = Agent("openai:gpt-4o", output_type=str)

@dataclass
class AskQuestion(BaseNode[ConvoState]):
    async def run(self, ctx: GraphRunContext[ConvoState]) -> EvalAnswer:
        result = await ask_agent.run(
            "Ask a question",
            message_history=ctx.state.messages,
        )
        ctx.state.messages += result.new_messages()
        return EvalAnswer(result.output)
```

### Logfire Observability

OpenTelemetry-based observability for AI applications. Always call `logfire.configure()` before any `instrument_*` calls.

**Setup & custom spans:**

```python
import logfire
from logfire_api import Logfire

logfire.configure(send_to_logfire="if-token-present")

# Scoped logger for your module
_logfire = Logfire(otel_scope="my-agent-framework")

# Custom spans
async def process_request(request: Request) -> Response:
    with _logfire.span("process_request", request_id=request.id):
        return await do_processing(request)

# Logging
logfire.info("User {user_id} logged in", user_id=123)
logfire.warn("Rate limit approaching", usage=0.95)
logfire.error("Request failed", error=str(e))
```

**LLM instrumentation:**

```python
import logfire

logfire.configure()

# Pydantic AI (traces agent runs, tool calls, retries, LLM interactions)
logfire.instrument_pydantic_ai()

# OpenAI (chat completions, embeddings, images, streaming)
logfire.instrument_openai()           # All clients
logfire.instrument_openai(client)     # Specific client

# Anthropic (messages, streaming, tool use, Bedrock)
logfire.instrument_anthropic()        # All clients
logfire.instrument_anthropic(client)  # Specific client

# OpenAI Agents SDK
logfire.instrument_openai_agents()
```

**Web & HTTP instrumentation:**

```python
from fastapi import FastAPI
import logfire

app = FastAPI()
logfire.configure()

# FastAPI (request spans with validated args, errors, timing)
logfire.instrument_fastapi(app)
logfire.instrument_fastapi(app, extra_spans=True)  # Arg parsing + endpoint spans

# HTTPX (outgoing HTTP calls)
logfire.instrument_httpx()
logfire.instrument_httpx(
    capture_headers=True,
    capture_request_body=True,
    capture_response_body=True,
)

# Requests (sync HTTP)
logfire.instrument_requests()
```

**Database & task queue instrumentation:**

```python
from sqlalchemy import create_engine
import logfire

logfire.configure()

# SQLAlchemy / SQLModel
engine = create_engine("postgresql://...")
logfire.instrument_sqlalchemy(engine=engine)

# Redis (sync + async)
logfire.instrument_redis(capture_statement=False)  # Default: hide commands

# Celery (call in BOTH worker and enqueuing app)
logfire.instrument_celery()
```

**Pydantic validation instrumentation:**

```python
import logfire

logfire.configure()

# Instrument Pydantic model validation (define models AFTER this call)
logfire.instrument_pydantic(
    include={"openai"},       # Include third-party modules
    exclude={"app.api.v1"},   # Exclude your modules
)

# Per-model config via plugin_settings
from logfire.integrations.pydantic import PluginSettings

class User(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"})):
    name: str
# record options: "all" | "failure" | "metrics" | "off"
```

**Pytest integration:**

```bash
pytest --logfire                              # Enable tracing
pytest --logfire-service-name my-service      # Custom service name
pytest --no-logfire                           # Disable when auto-enabled
```

```python
# Test fixture
def test_workflow(logfire_pytest):
    with logfire_pytest.span("create user"):
        user = create_user()
    logfire_pytest.info("User created", user_id=user.id)
```

**Key patterns:**

| Pattern | Usage |
|---------|-------|
| Global vs instance | `instrument_X()` = all instances; `instrument_X(client)` = one instance |
| Service names | `logfire.configure(service_name="worker")` to distinguish services |
| Streaming | LLM integrations create 2 spans: request + stream consumption |
| Distributed tracing | Call `instrument_celery()` in both producer and consumer |
| Security | Sensitive data not captured by default; use hooks for control |

### Abstract Store Interface

```python
class AgentStore(ABC):
    @abstractmethod
    async def create_agent(self, name: str) -> Agent: ...
    @abstractmethod
    async def read_agent(self, agent_id: AgentId) -> Agent: ...
    @abstractmethod
    async def list_agents(self, tag_id: TagId | None = None) -> Sequence[Agent]: ...
    @abstractmethod
    async def update_agent(self, agent_id: AgentId, params: AgentUpdateParams) -> Agent: ...
    @abstractmethod
    async def delete_agent(self, agent_id: AgentId) -> None: ...
```

### Pydantic Evals

Systematic testing and evaluation of agent/LLM outputs.

**Dataset & Cases:**

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected, IsInstance, LLMJudge

dataset = Dataset(
    cases=[
        Case(
            name="capital_question",
            inputs="What is the capital of France?",
            expected_output="Paris",
            metadata={"difficulty": "easy"},
        ),
        Case(
            name="math_question",
            inputs="What is 2+2?",
            expected_output="4",
            evaluators=[Contains("4")],  # Case-level evaluator
        ),
    ],
    evaluators=[  # Dataset-level evaluators (run on all cases)
        IsInstance(type_name="str"),
        EqualsExpected(),
    ],
)
```

**Running evaluations:**

```python
# Async
report = await dataset.evaluate(my_async_task)

# Sync
report = dataset.evaluate_sync(my_sync_task)

# Print report
report.print(include_input=True, include_output=True)

# Compare against baseline
report.print(baseline=baseline_report)
```

**Built-in evaluators:**

```python
from pydantic_evals.evaluators import (
    EqualsExpected,     # output == expected_output
    Equals,             # output == specific value
    Contains,           # substring/membership check
    IsInstance,         # type check
    MaxDuration,        # performance threshold
    LLMJudge,          # LLM-based assessment
    HasMatchingSpan,   # OpenTelemetry span check (requires logfire)
)

# LLMJudge with score output
LLMJudge(
    rubric="Response is factually accurate and well-structured",
    model="anthropic:claude-sonnet-4-6",
    include_input=True,
    include_expected_output=True,
)

# LLMJudge with numeric score (instead of pass/fail)
LLMJudge(
    rubric="Overall response quality",
    score={"evaluation_name": "quality", "include_reason": True},
    assertion=False,
)

# Performance check
MaxDuration(seconds=2.0)
```

**Custom evaluators:**

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

@dataclass
class WordCount(Evaluator):
    min_words: int = 10

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        count = len(str(ctx.output).split())
        return EvaluationReason(
            value=count >= self.min_words,
            reason=f"Word count: {count}, minimum: {self.min_words}",
        )

# Evaluator returning multiple metrics
@dataclass
class QualityCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float | str]:
        return {
            "has_content": len(ctx.output) > 0,
            "length_score": min(len(ctx.output) / 100, 1.0),
            "category": "short" if len(ctx.output) < 50 else "long",
        }

# Async evaluator
@dataclass
class APICheck(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={"output": ctx.output})
            return resp.json()["valid"]
```

**EvaluatorContext fields:**

```python
ctx.name              # Case name
ctx.inputs            # Case inputs
ctx.output            # Task output
ctx.expected_output   # Expected output (if set)
ctx.metadata          # Case metadata
ctx.duration          # Execution duration (seconds)
ctx.metrics           # dict[str, int | float]
ctx.attributes        # dict[str, Any]
ctx.span_tree         # OpenTelemetry spans (if logfire)
```

**Dataset persistence (YAML/JSON):**

```python
# Save
dataset.to_file("evals.yaml")
dataset.to_file("evals.json")

# Save with custom evaluator types
dataset.to_file("evals.yaml", custom_evaluator_types=[WordCount])

# Load
dataset = Dataset[str, str, Any].from_file("evals.yaml")
dataset = Dataset[str, str, Any].from_file(
    "evals.yaml", custom_evaluator_types=[WordCount],
)
```

**Best practices:**

```python
# Combine fast deterministic + expensive LLM evaluators
dataset = Dataset(
    cases=[Case(inputs="test", expected_output="result")],
    evaluators=[
        IsInstance(type_name="str"),           # Fast: fail early
        Contains(value="keyword"),             # Fast: substring check
        MaxDuration(seconds=2.0),              # Fast: performance
        LLMJudge(rubric="Accurate response"),  # Expensive: last
    ],
)

# Use temperature=0 for LLMJudge consistency
from pydantic_ai import ModelSettings

LLMJudge(
    rubric="Accurate response",
    model_settings=ModelSettings(temperature=0.0),
)
```

---

## 7. Pydantic Validation

Data validation, serialization, and model configuration patterns.

### Rules

- **Prefer `Annotated` validators** over class-method decorators for reusability
- **Use `Field()` constraints** over custom validators for simple bounds checking
- **Use `model_config = ConfigDict(...)** over inner `class Config`
- **Use `model_validate()`** over direct constructor for external/untrusted data
- **Use `TypeAdapter`** for validating non-model types (lists, dicts, primitives)
- **Use `@validate_call`** for function argument validation
- **Use discriminated unions** for polymorphic models (performance + clear errors)

### Field Constraints

```python
from pydantic import BaseModel, Field
from decimal import Decimal

class Model(BaseModel):
    positive: int = Field(gt=0)
    non_negative: int = Field(ge=0)
    limited: int = Field(lt=100)
    short_str: str = Field(min_length=1, max_length=255)
    pattern: str = Field(pattern=r"^[a-z]+$")
    precise: Decimal = Field(max_digits=5, decimal_places=2)
    multiple: int = Field(multiple_of=5)
```

### Field Configuration

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(alias="user_name")
    age: int = Field(validation_alias="user_age")
    email: str = Field(serialization_alias="email_address")
    password: str = Field(exclude=True)
    tags: list[str] = Field(default_factory=list)
    secret: str = Field(repr=False)
    created_at: str = Field(frozen=True)
    legacy: int = Field(deprecated="Use new_field")
```

### Model Configuration

```python
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(
        strict=True,                    # No type coercion
        frozen=True,                    # Immutable instances
        validate_assignment=True,       # Validate on attr set
        validate_default=True,          # Validate defaults
        from_attributes=True,           # Accept ORM objects
        str_strip_whitespace=True,      # Strip string fields
        str_max_length=1000,            # Global string limit
        use_attribute_docstrings=True,  # Docstrings as descriptions
    )
```

### Field Validators

```python
from pydantic import BaseModel, field_validator

class Model(BaseModel):
    name: str
    age: int

    @field_validator("name", mode="before")  # Raw input → before parsing
    @classmethod
    def normalize_name(cls, v: Any) -> Any:
        return v.strip().title() if isinstance(v, str) else v

    @field_validator("age", mode="after")  # After parsing → typed value
    @classmethod
    def check_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError("age must be non-negative")
        return v
```

### Model Validators

```python
from pydantic import BaseModel, model_validator
from typing_extensions import Self

class UserCreate(BaseModel):
    password: str
    password_confirm: str

    @model_validator(mode="after")
    def passwords_match(self) -> Self:
        if self.password != self.password_confirm:
            raise ValueError("passwords do not match")
        return self

    @model_validator(mode="before")
    @classmethod
    def strip_sensitive(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("card_number", None)
        return data
```

### Annotated Validators (Preferred for Reuse)

```python
from typing import Annotated
from pydantic import AfterValidator, BeforeValidator, WrapValidator, BaseModel

def ensure_list(v: Any) -> Any:
    return [v] if not isinstance(v, list) else v

def is_even(v: int) -> int:
    if v % 2 != 0:
        raise ValueError("must be even")
    return v

EvenInt = Annotated[int, AfterValidator(is_even)]
FlexList = Annotated[list[int], BeforeValidator(ensure_list)]

class Model(BaseModel):
    even: EvenInt              # Reusable across models
    numbers: FlexList
    items: list[EvenInt]       # Composable in containers
```

### Serialization

```python
from pydantic import BaseModel, computed_field, field_serializer

class Box(BaseModel):
    width: float
    height: float
    depth: float

    @computed_field
    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @field_serializer("width", "height", "depth")
    def round_dims(self, v: float) -> float:
        return round(v, 2)

# model_dump options
m.model_dump()                       # dict
m.model_dump_json()                  # JSON string
m.model_dump(by_alias=True)          # Use aliases
m.model_dump(exclude={"password"})   # Exclude fields
m.model_dump(exclude_none=True)      # Drop None values
m.model_dump(exclude_unset=True)     # Only explicitly set
m.model_dump(exclude_defaults=True)  # Drop defaults
```

### Discriminated Unions

```python
from typing import Literal, Union
from pydantic import BaseModel, Field, Discriminator, Tag
from typing import Annotated

# String discriminator (preferred when models share a type field)
class Cat(BaseModel):
    pet_type: Literal["cat"]
    meows: int

class Dog(BaseModel):
    pet_type: Literal["dog"]
    barks: float

class Owner(BaseModel):
    pet: Union[Cat, Dog] = Field(discriminator="pet_type")

# Callable discriminator (when no shared field)
def get_food_type(v: Any) -> str:
    if isinstance(v, dict):
        return "fruit" if "fruit" in v else "vegetable"
    return getattr(v, "fruit", "vegetable")

class Meal(BaseModel):
    food: Annotated[
        Union[
            Annotated[Apple, Tag("fruit")],
            Annotated[Carrot, Tag("vegetable")],
        ],
        Discriminator(get_food_type),
    ]
```

### Strict Mode

```python
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

# Model-level strict (no "42" → 42 coercion)
class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True)
    name: str
    age: int = Field(strict=False)  # Override per-field

# Field-level strict via types
class MixedModel(BaseModel):
    name: StrictStr
    count: StrictInt
    flexible: int           # Allows coercion
```

### TypeAdapter (Non-Model Validation)

```python
from pydantic import TypeAdapter

adapter = TypeAdapter(list[dict[str, int]])
result = adapter.validate_python([{"id": "123"}])       # Coerces
result = adapter.validate_json('[{"id": "123"}]')        # From JSON
json_bytes = adapter.dump_json(result)                   # To JSON
schema = adapter.json_schema()                           # JSON Schema
```

### Validation Methods

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# From dict
user = User.model_validate({"name": "John", "age": "30"})

# From JSON
user = User.model_validate_json('{"name": "John", "age": 30}')

# Strict (no coercion)
user = User.model_validate(data, strict=True)

# From ORM/attrs (requires from_attributes=True)
user = User.model_validate(orm_obj, from_attributes=True)

# With context (accessible in validators)
user = User.model_validate(data, context={"role": "admin"})
```

### Function Validation

```python
from pydantic import validate_call
from datetime import date

@validate_call
def repeat(s: str, count: int, *, separator: bytes = b"") -> bytes:
    return separator.join(s.encode() for _ in range(count))

repeat("hello", "3")  # count coerced str → int

@validate_call(validate_return=True)
def get_date(d: date) -> date:
    return d
```

### Error Handling

```python
from pydantic import BaseModel, ValidationError, PydanticCustomError, field_validator

class Model(BaseModel):
    x: int

    @field_validator("x")
    @classmethod
    def validate_x(cls, v: int) -> int:
        if v > 100:
            raise PydanticCustomError(
                "value_too_large",
                "Value must be <= {max}",
                {"max": 100},
            )
        return v

try:
    Model(x="invalid")
except ValidationError as e:
    e.errors()       # list[dict] of error details
    e.error_count()  # int
    e.json()         # JSON string
```

### Generic Models

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class Response(BaseModel, Generic[T]):
    data: T
    count: int

# Parameterized usage
Response[list[str]](data=["a", "b"], count=2)
```

### Aliases & ORM Integration

```python
from pydantic import BaseModel, ConfigDict, Field, AliasPath, AliasChoices

class APIResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    first_name: str = Field(validation_alias=AliasPath("names", 0))
    phone: str = Field(
        validation_alias=AliasChoices("phone", "tel", "phone_number"),
    )

# Accepts nested/aliased input
APIResponse.model_validate({"names": ["John"], "tel": "555-1234"})

# ORM objects work directly
user_model = APIResponse.model_validate(orm_instance)
```

---

## 8. Project Configuration

Standards for project setup and tooling.

### Rules

- **uv** as package manager
- `.python-version` file in project root (e.g., `3.13.1`)
- `mkdocs.yml` for documentation (Material theme)
- `AGENTS.md` or `CLAUDE.md` for AI assistant guidance
- Comprehensive `README.md` with badges, architecture, quick start, structure

### uv Commands

```bash
uv sync                    # Install dependencies
uv add pydantic-ai         # Add dependency
uv add --dev pytest        # Add dev dependency
uv run pytest -v           # Run in venv
uv run python script.py    # Run script
```

### Code Quality

```bash
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run mypy .              # Type check
```

---

## Resources

**Framework Docs**: [Pydantic](https://docs.pydantic.dev/) • [Pydantic AI](https://ai.pydantic.dev/) • [Pydantic Graph](https://ai.pydantic.dev/graph/) • [Pydantic Evals](https://ai.pydantic.dev/evals/) • [Logfire](https://logfire.pydantic.dev/)

**Style Guides**: [PEP 8](https://peps.python.org/pep-0008/) • [PEP 257](https://peps.python.org/pep-0257/) • [Google Python Style](https://google.github.io/styleguide/pyguide.html)

**Tools**: [Ruff](https://docs.astral.sh/ruff/) • [mypy](https://mypy.readthedocs.io/) • [Pyright](https://microsoft.github.io/pyright/) • [pytest](https://docs.pytest.org/)
