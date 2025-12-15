# Installation

AGENT-K requires Python 3.11+ and uses [uv](https://github.com/astral-sh/uv) for dependency management.

## Prerequisites

- **Python 3.11+** — [Download Python](https://www.python.org/downloads/)
- **uv** — Fast Python package manager
- **Node.js 20+** — For the frontend (optional)
- **pnpm** — For frontend dependencies (optional)

### Install uv

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "pip"

    ```bash
    pip install uv
    ```

## Backend Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mikewcasale/agent-k.git
cd agent-k
```

### 2. Install Dependencies

```bash
cd backend
uv sync
```

This creates a virtual environment in `.venv` and installs all dependencies.

### 3. Activate the Environment

=== "macOS / Linux"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows"

    ```powershell
    .venv\Scripts\activate
    ```

### 4. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# Kaggle API (required)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Model providers (at least one required)
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...

# Optional: Local LM Studio endpoint
DEVSTRAL_BASE_URL=http://192.168.105.1:1234/v1

# Optional: Observability
LOGFIRE_TOKEN=...
```

### Getting Kaggle Credentials

1. Log in to [Kaggle](https://www.kaggle.com/)
2. Go to **Account** → **API** → **Create New Token**
3. This downloads `kaggle.json` with your credentials

## Frontend Installation (Optional)

The frontend provides a real-time mission monitoring dashboard.

```bash
cd frontend
pnpm install
```

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:9000
AUTH_SECRET=your-secret-key
```

## Verify Installation

```bash
cd backend
uv run python -c "from agent_k import LycurgusOrchestrator; print('✓ AGENT-K installed')"
```

## Development Installation

For development with linting and testing tools:

```bash
uv sync --all-extras
```

This installs the `dev` and `docs` optional dependencies.

## Running the Servers

### Backend Only

```bash
cd backend
source .venv/bin/activate
python -m agent_k.ui.ag_ui
```

The API server runs at `http://localhost:9000`.

### Frontend Only

```bash
cd frontend
pnpm dev
```

The dashboard runs at `http://localhost:3000`.

### Both Servers

From the project root:

```bash
./run.sh
```

This starts both servers and handles cleanup on exit.

## Next Steps

- [Quick Start](quick-start.md) — Run your first mission
- [Concepts](concepts/agents.md) — Understand the architecture
- [Examples](examples/multi-agent-demo.md) — See full examples

