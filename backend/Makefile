# AGENT-K Backend Makefile

.PHONY: help install dev test lint format mypy check docs docs-serve clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

dev:  ## Install all dependencies including dev and docs
	uv sync --all-extras

test:  ## Run tests
	uv run pytest -v

test-cov:  ## Run tests with coverage
	uv run pytest --cov=agent_k --cov-report=html --cov-report=term

lint:  ## Run linting
	uv run ruff check .

format:  ## Format code
	uv run ruff format .

mypy:  ## Run type checking
	uv run mypy .

check: lint mypy test  ## Run all checks (lint, mypy, test)

docs:  ## Build documentation
	uv run mkdocs build

docs-serve:  ## Serve documentation locally
	uv run mkdocs serve

docs-deploy:  ## Deploy documentation to GitHub Pages
	uv run mkdocs gh-deploy

clean:  ## Clean build artifacts
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf site/
	rm -rf htmlcov/
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:  ## Run the API server
	uv run python -m agent_k.ui.ag_ui

demo:  ## Run the multi-agent demo
	uv run python examples/multi_agent_playbook.py --model devstral:local

