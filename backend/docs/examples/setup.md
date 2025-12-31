# Example Prerequisites

Before running the examples, ensure you have completed the setup.

## Requirements

### Backend

1. **Python 3.11+** and uv installed
2. **Dependencies installed**: `uv sync` in `backend/`
3. **Environment configured**: `.env` file with API keys

### Environment Variables

Create `backend/.env`:

```bash
# Required: Kaggle API
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Required: At least one model provider
ANTHROPIC_API_KEY=sk-ant-...
# OR
OPENROUTER_API_KEY=sk-or-v1-...
# OR
OPENAI_API_KEY=sk-...

# Optional: Local Devstral
DEVSTRAL_BASE_URL=http://localhost:1234/v1
```

### Model Options

Examples support multiple models:

| Model Spec | Requirements |
|------------|--------------|
| `devstral:local` | LM Studio running with Devstral |
| `anthropic:claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` |
| `openrouter:mistralai/devstral-small` | `OPENROUTER_API_KEY` |
| `openai:gpt-4o` | `OPENAI_API_KEY` |

## Verify Setup

```bash
cd backend
uv run python -c "
from agent_k.infra.providers import get_model
print('✓ agent_k importable')

from agent_k.toolsets import create_memory_backend, prepare_web_search
print('✓ tool helpers available')
"
```

## Running Examples

All examples are in `backend/examples/`. Use the same playbook for any model provider by passing `--model`:

```bash
cd backend

# Multi-agent playbook (built-in tools)
uv run python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307

# Multi-agent playbook (Devstral local)
uv run python examples/multi_agent_playbook.py --model devstral:local
```

## Local Devstral Setup

For local model inference:

1. **Install LM Studio** from [lmstudio.ai](https://lmstudio.ai/)
2. **Download Devstral**: Search for `mistralai/devstral-small-2-2512`
3. **Start Local Server**: Click "Start Server" in LM Studio
4. **Verify**: `curl http://localhost:1234/v1/models`

The default endpoint is `http://192.168.105.1:1234/v1`. Override with:

```bash
export DEVSTRAL_BASE_URL=http://localhost:1234/v1
```

## Next Steps

- [Multi-Agent Demo](multi-agent-demo.md) — Full walkthrough
- [Custom Agent](custom-agent.md) — Create your own agent
