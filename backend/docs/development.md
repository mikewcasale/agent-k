# Development

All Python commands use `backend/.venv/bin/python` with Python 3.13. Backend modules are flat,
lowercase, and single-word except `__init__.py` and `__main__.py`. Code uses 100-column formatting,
Google-style docstrings, strict typing, explicit resource lifecycles, and Logfire observability.

## Required checks

```bash
backend/.venv/bin/python -m ruff format --check backend
backend/.venv/bin/python -m ruff check backend
backend/.venv/bin/python -m mypy --config-file backend/pyproject.toml backend/agent_k backend/tools
backend/.venv/bin/python -m pytest -v backend/tests
cd frontend && pnpm lint && pnpm build
cd backend && .venv/bin/python -m mkdocs build --strict
```

Formatting fixes run before checks. Re-run checks after a formatter changes files.

## Test policy

Tests must exercise production parsers, DTOs, algorithms, stores, and service boundaries. Do not
use mocks, monkeypatches, fake service clients, fabricated competition payloads, or synthetic ML
rows to make a path appear validated.

- Pure structural tests may inspect OpenAPI, schemas, source boundaries, and event registries.
- Deterministic algorithm tests use checked-in, provenance-documented datasets.
- PostgreSQL, S3, Kaggle, model-provider, and end-to-end tests run only when their real configured
  services are present; otherwise they skip with an explicit reason.
- A passing local structural suite is not production, deployment, or Kaggle acceptance.

## Adding code

Prefer an existing flat module. New modules must represent a distinct domain and remain free of
import-time filesystem, network, credential, database, model, or telemetry initialization. API
routes keep public camelCase fields and stable event names. Changes to job, snapshot, result,
artifact, or event contracts require coordinated updates to API DTOs, worker clients, persistence,
frontend types, migration handling, tests, and these docs.
