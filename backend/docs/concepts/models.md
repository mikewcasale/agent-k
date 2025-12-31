# Model Configuration

AGENT-K supports multiple model providers through a unified `get_model()` factory function. This allows you to switch between providers without changing agent code.

## Supported Models

| Model Spec | Provider | Description |
|------------|----------|-------------|
| `devstral:local` | Local LM Studio | Devstral running locally |
| `devstral:http://host:port/v1` | Custom endpoint | Custom LM Studio server |
| `anthropic:claude-3-haiku-20240307` | Anthropic | Claude Haiku |
| `anthropic:claude-sonnet-4-20250514` | Anthropic | Claude Sonnet |
| `openrouter:mistralai/devstral-small` | OpenRouter | Devstral via OpenRouter |
| `openai:gpt-4o` | OpenAI | GPT-4o |

## Using get_model()

The `get_model()` function resolves model specifications:

```python
from agent_k.infra.providers import get_model

# Standard Pydantic-AI model string (passed through)
model = get_model('anthropic:claude-3-haiku-20240307')

# Local Devstral (returns OpenAIChatModel)
model = get_model('devstral:local')

# OpenRouter (returns OpenAIChatModel)
model = get_model('openrouter:mistralai/devstral-small')
```

## Local Devstral (LM Studio)

To use Devstral locally:

### 1. Install LM Studio

Download from [lmstudio.ai](https://lmstudio.ai/)

### 2. Load Devstral Model

In LM Studio:
- Download `mistralai/devstral-small-2-2512`
- Start the local server

### 3. Configure Endpoint

=== "Default (192.168.105.1:1234)"

    ```python
    model = get_model('devstral:local')
    ```

=== "Custom URL"

    ```python
    model = get_model('devstral:http://localhost:1234/v1')
    ```

=== "Environment Variable"

    ```bash
    export DEVSTRAL_BASE_URL=http://localhost:1234/v1
    ```

    ```python
    model = get_model('devstral:local')  # Uses env var
    ```

## Anthropic (Claude)

### Setup

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)
2. Set the environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Available Models

```python
# Claude Haiku (fast, cost-effective)
model = get_model('anthropic:claude-3-haiku-20240307')

# Claude Sonnet (balanced)
model = get_model('anthropic:claude-sonnet-4-20250514')

# Claude Opus (most capable)
model = get_model('anthropic:claude-opus-4-20250514')
```

## OpenRouter

OpenRouter provides access to many models through a unified API.

### Setup

1. Get an API key from [openrouter.ai](https://openrouter.ai/)
2. Set the environment variable:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Available Models

```python
# Devstral (tool-capable coding model)
model = get_model('openrouter:mistralai/devstral-small')

# Free tier
model = get_model('openrouter:mistralai/devstral-2512:free')

# Other models
model = get_model('openrouter:anthropic/claude-3-haiku')
model = get_model('openrouter:openai/gpt-4o')
```

## OpenAI

### Setup

1. Get an API key from [platform.openai.com](https://platform.openai.com/)
2. Set the environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

### Available Models

```python
model = get_model('openai:gpt-4o')
model = get_model('openai:gpt-4o-mini')
model = get_model('openai:gpt-4-turbo')
```

## Using Models with Agents

Agents are module-level singletons. Configure their model via environment variables
before importing them:

```bash
# With Anthropic
export LOBBYIST_MODEL=anthropic:claude-3-haiku-20240307

# With local Devstral
export LOBBYIST_MODEL=devstral:local

# With OpenRouter
export LOBBYIST_MODEL=openrouter:mistralai/devstral-small
```

```python
from agent_k.agents.lobbyist import lobbyist_agent

result = await lobbyist_agent.run('Find featured competitions', deps=deps)
```

## Using Models with Orchestrator

```python
from agent_k.agents.lycurgus import LycurgusOrchestrator, LycurgusSettings

# Via config
config = LycurgusSettings(
    default_model='anthropic:claude-3-haiku-20240307',
)
orchestrator = LycurgusOrchestrator(config=config)

# Via model parameter
orchestrator = LycurgusOrchestrator(model='devstral:local')

# Devstral helper
config = LycurgusSettings.with_devstral(
    base_url='http://localhost:1234/v1'  # Optional
)
```

## Model Implementation Details

### OpenAI-Compatible Models

For `devstral:` and `openrouter:` specs, `get_model()` returns an `OpenAIChatModel`:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

def create_devstral_model(base_url: str | None = None) -> OpenAIChatModel:
    return OpenAIChatModel(
        'mistralai/devstral-small-2-2512',
        provider=OpenAIProvider(
            base_url=base_url or DEVSTRAL_BASE_URL,
            api_key='not-required',  # Local doesn't need auth
        ),
    )
```

### Standard Models

For standard specs like `anthropic:...`, the string is passed directly to pydantic-ai which handles model resolution.

## Checking Model Type

```python
from agent_k.infra.providers import is_devstral_model

# Check if using Devstral
if is_devstral_model('devstral:local'):
    print("Using local Devstral")
```

## Best Practices

### Development

Use local Devstral for development to:
- Avoid API costs
- Work offline
- Faster iteration

```python
# In development
model = get_model('devstral:local')
```

### Production

Use Anthropic or OpenRouter for production:

```python
# In production
model = get_model('anthropic:claude-3-haiku-20240307')
```

### Testing

Use `output_type=str` for broad model compatibility:

```python
agent = Agent(
    model,
    output_type=str,  # Works with all models
    retries=5,        # More retries for less capable models
)
```

## Next Steps

- [Agents](agents.md) — How agents use models
- [Toolsets](toolsets.md) — Tools that work with any model
- [Quick Start](../quick-start.md) — Try different models
