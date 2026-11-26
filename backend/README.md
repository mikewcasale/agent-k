# AGENT-K backend

The backend is a flat Python 3.13 package with two deployment roles:

- `agent_k.api` is the FastAPI control plane for authenticated frontend and worker traffic.
- `agent_k.worker` is the portable ML runtime that claims fenced jobs and runs each mission in a
  fresh isolated subprocess and sandbox.

PostgreSQL persists missions, controls, jobs, leases, ordered events, snapshots, results, artifact
metadata, and migration ledgers. S3 stores immutable artifacts. The API owns both credentials;
workers use only authenticated HTTP and short-lived signed storage URLs.

## Install

```bash
uv sync --all-extras
```

The `worker` extra contains numerical and model runtimes. The `docs` extra contains the locked
MkDocs toolchain. All commands use `.venv/bin/python`.

## Entry points

```bash
# API
.venv/bin/python -m uvicorn agent_k.api:app --host 0.0.0.0 --port 9000

# Worker
.venv/bin/python -m agent_k.worker

# Package version
.venv/bin/python -m agent_k
```

The API requires `AGENT_K_DATABASE_URL`, `AGENT_K_GATEWAY_SECRET`, and
`AGENT_K_WORKER_TOKEN`. S3-backed artifact and migration routes require `AGENT_K_S3_BUCKET`.
Kaggle routes return an explicit unavailable response when `KAGGLE_USERNAME` and `KAGGLE_KEY`
are absent.

## Package map

| Area | Flat modules |
| --- | --- |
| Control plane | `api`, `config`, `events`, `intent`, `observability` |
| Mission | `lycurgus`, `mission`, `lobbyist`, `scientist`, `tracking`, `evolution` |
| Runtime | `stores`, `runner`, `worker`, `sandbox`, `migration` |
| Data and evaluation | `data`, `datasets`, `evals`, `evaluator`, `hints`, `losses`, `metrics` |
| Model families | `tabular`, `text`, `vision` |
| Integrations and domain | `kaggle`, `providers`, `search`, `memory`, `interfaces`, `models`, `types`, `exceptions` |

No module creates a database connection, reads credentials, configures telemetry, accesses files,
or instantiates a model at import time. API lifespan and worker/mission entry points own those
resources explicitly.

## Durable contracts

- Public JSON and browser event fields remain camelCase.
- Worker mutations require a live lease ID and matching worker identity.
- Event appends are contiguous and idempotent; SSE replay uses a mission-global sequence.
- Pause checkpoints before interruption; resume restores the checkpoint and verified artifacts.
- Terminal job commits require the matching `mission-complete` event.
- Artifact and migration uploads remain unavailable until S3 size and SHA-256 verification.
- Saved legacy state is imported only through the explicit content-addressed migration ledger.

## Checks

From the repository root:

```bash
backend/.venv/bin/python -m ruff format --check backend
backend/.venv/bin/python -m ruff check backend
backend/.venv/bin/python -m mypy --config-file backend/pyproject.toml backend/agent_k
backend/.venv/bin/python -m pytest -v backend/tests
cd backend && .venv/bin/python -m mkdocs build --strict
```

Backend code uses 100-column formatting, Google-style docstrings, strict typing, Logfire, and the
repository's no-mock/no-synthetic-data test policy.

See the [MkDocs overview](docs/index.md) for installation, architecture, APIs, operations,
migration, and generated module references.
