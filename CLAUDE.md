# AGENT-K working agreements

AGENT-K runs Kaggle missions through discovery, research, prototype, evolution,
and submission. Use Python 3.13 through `backend/.venv/bin/python`.

## Canonical Python guide

Follow `.context/styleguides/python-ai-style-guide-compressed.md`. Backend modules
are flat single lowercase words under `backend/agent_k`, except Python dunder
entry points. Related types, configuration, tools, and implementations live
together. Use relative imports, 100-column Ruff formatting, Google docstrings,
MIT module notices, aliased future annotations, explicit `__all__` tuples,
modern typing, and `TYPE_CHECKING` for annotation-only imports. SAGE is retired.

No placeholders, stubs, mocked integrations, synthetic project datasets, or
invented success/results. Use real datasets and report unexecuted service gates.
The approved fixed NVIDIA Kumo pretrained weights are permitted even though
their vendor training corpus includes synthetic tables.

## Runtime architecture

Render hosts a lightweight FastAPI API using dedicated PostgreSQL runtime state
and AWS S3 artifacts. Portable Linux workers poll authenticated API endpoints.
Run each mission in a fresh process. Run every candidate inside the rootless
Docker sandbox with no network, secrets, hidden labels, or host socket.

OpenEvolve is the sole evolution engine. LightGBM is the mandatory tabular baseline
and its real gradient/Hessian custom objectives must be evaluated. Kumo is an
optional fixed-weight challenger for single-target tabular tasks. Never replace
measured local fitness with public scores or infer rank from a matching score.

Support generic tabular, vision, and text classification/regression; never add
competition-specific modules, data, code, or configurations. Keep HTTP routes,
camelCase payloads, and existing SSE event names stable. New iteration fields
coexist with deprecated generation aliases.

Agent singletons use `Agent(model=None, ...)`; providers resolve only at execution.
Use Pydantic AI, Pydantic Graph, Pydantic Evals, and Logfire public APIs before
building custom replacements. Observability configuration belongs in lifespan
or process entry points. Do not initialize providers, files, or services on import.

## Validation

From the repository root:

```bash
backend/.venv/bin/python -m ruff format --check backend/agent_k backend/tests
backend/.venv/bin/python -m ruff check backend/agent_k backend/tests
cd backend
.venv/bin/python -m mypy agent_k
.venv/bin/python -m pytest -v --tb=short
.venv/bin/python -m mkdocs build --strict
```

Use `uv sync --locked --extra worker --extra kumo --extra docs` for the complete
development environment; API installs omit the ML extras. Run frontend `pnpm lint`
and `pnpm build` for frontend changes. Before committing, rerun any checks affected
by formatter or linter changes. A passing local test is not a deployment claim.
Do not commit, push, deploy, provision infrastructure, or submit to Kaggle unless
the user has authorized that action. Submission is disabled by default.

## Subagents

Every spawned subagent must explicitly use model `gpt-5.6-sol`, reasoning effort
`xhigh`, and `fork_turns: none`. If unavailable, continue in the primary agent.
