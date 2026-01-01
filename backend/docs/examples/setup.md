# Example Prerequisites

Before running the example snippets in the docs, ensure you have completed the setup.

## Requirements

### Backend

1. Python 3.11+ and uv installed
2. Dependencies installed with `uv sync` in `backend/`
3. Environment configured with API keys

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

| Model Spec | Requirements |
|------------|--------------|
| `devstral:local` | LM Studio running with Devstral |
| `anthropic:claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` |
| `openrouter:mistralai/devstral-small-2-2512` | `OPENROUTER_API_KEY` |
| `openai:gpt-4o` | `OPENAI_API_KEY` |

## Verify Setup

```bash
cd backend
uv run python -c "from agent_k import LycurgusOrchestrator; print('ok')"
```

## Running Example Snippets

The examples in this section can be run by creating a small script or using a heredoc:

```bash
cd backend
uv run python - <<'PY'
print("Agent-K ready")
PY
```

## Next Steps

- [Multi-Agent Demo](multi-agent-demo.md) - Walkthrough using the core agents
- [Custom Agent](custom-agent.md) - Create your own agent
