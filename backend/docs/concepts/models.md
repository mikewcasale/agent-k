# Model Configuration

AGENT-K supports multiple model providers through a unified `get_model()` factory. Standard Pydantic-AI model strings are passed through, while `devstral:` and `openrouter:` specs are resolved to OpenAI-compatible models.

## Common Model Specs

| Model Spec | Provider | Description |
|------------|----------|-------------|
| `devstral:local` | Local LM Studio | Devstral running locally (uses `DEVSTRAL_BASE_URL`) |
| `devstral:http://host:port/v1` | Custom endpoint | Custom LM Studio server |
| `devstral:mistralai/devstral-small-2-2512` | Local LM Studio | Override the model id |
| `anthropic:claude-3-haiku-20240307` | Anthropic | Claude Haiku |
| `anthropic:claude-sonnet-4-5` | Anthropic | Claude Sonnet (backend default) |
| `openrouter:mistralai/devstral-small-2-2512` | OpenRouter | Devstral via OpenRouter |
| `openai:gpt-4o` | OpenAI | GPT-4o |
| `openrouter:a:free,openrouter:b:free` | Any | Failover chain tried left to right |

`get_model()` will accept any standard Pydantic-AI model string for supported providers.

## Failover Chains

A spec may list several models separated by commas. `get_model()` resolves such a spec to a
pydantic-ai `FallbackModel`, which tries each member in order and moves on whenever a provider
raises a `ModelAPIError` (rate limits, 5xx, dead model slugs, refused connections). Only when every
member fails does the run see a `FallbackExceptionGroup`.

```python
from agent_k.infra.providers import get_model

# Fails over to the smaller free model when the primary is rate limited.
model = get_model("openrouter:openai/gpt-oss-120b:free,openrouter:openai/gpt-oss-20b:free")
```

Chain members whose provider credentials are absent are skipped at construction time, so a mixed
chain such as `anthropic:claude-sonnet-4-5,openrouter:openai/gpt-oss-20b:free` still works when only
`OPENROUTER_API_KEY` is set. `DEFAULT_MODEL` ships as a chain, so every agent gets failover without
extra wiring, and `evolution_models` entries may themselves be chains.

## Using get_model()

```python
from agent_k.infra.providers import get_model

# Standard Pydantic-AI model string (passed through)
model = get_model("anthropic:claude-3-haiku-20240307")

# Local Devstral (returns OpenAIChatModel)
model = get_model("devstral:local")

# OpenRouter (returns OpenAIChatModel)
model = get_model("openrouter:mistralai/devstral-small-2-2512")
```

## Local Devstral (LM Studio)

To use Devstral locally:

1. Install LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Download `mistralai/devstral-small-2-2512`
3. Start the local server

The default endpoint is `http://192.168.105.1:1234/v1`. Override with:

```bash
export DEVSTRAL_BASE_URL=http://localhost:1234/v1
```

Then:

```python
model = get_model("devstral:local")
```

## Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
model = get_model("anthropic:claude-3-haiku-20240307")
model = get_model("anthropic:claude-sonnet-4-5")
```

## OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

```python
model = get_model("openrouter:mistralai/devstral-small-2-2512")
model = get_model("openrouter:anthropic/claude-3-haiku")
```

## OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

```python
model = get_model("openai:gpt-4o")
```

## Using Models with Agents

Configure agent models via environment variables before importing:

```bash
export LOBBYIST_MODEL=anthropic:claude-3-haiku-20240307
export SCIENTIST_MODEL=devstral:local
export EVOLVER_MODEL=openrouter:mistralai/devstral-small-2-2512
```

```python
from agent_k.agents.lobbyist import lobbyist_agent

result = await lobbyist_agent.run("Find featured competitions", deps=deps)
```

## Using Models with the Orchestrator

```python
from agent_k.agents.lycurgus import LycurgusOrchestrator, LycurgusSettings

# Via config
config = LycurgusSettings(
    default_model="anthropic:claude-3-haiku-20240307",
)
orchestrator = LycurgusOrchestrator(config=config)

# Via model parameter
orchestrator = LycurgusOrchestrator(model="devstral:local")

# Devstral helper
config = LycurgusSettings.with_devstral(
    base_url="http://localhost:1234/v1"  # Optional
)
```
