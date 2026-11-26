PYTHON := .venv/bin/python
.PHONY: install test lint format mypy typecheck check docs run
install:
	uv sync --locked --extra worker --extra kumo --extra docs
test:
	$(PYTHON) -m pytest -v --tb=short
lint:
	$(PYTHON) -m ruff check agent_k tests tools
format:
	$(PYTHON) -m ruff format agent_k tests tools
mypy:
	$(PYTHON) -m mypy agent_k tools
typecheck: mypy
check: lint mypy test
docs:
	$(PYTHON) -m mkdocs build --strict
run:
	$(PYTHON) -m uvicorn agent_k.api:app --host 0.0.0.0 --port 9000
