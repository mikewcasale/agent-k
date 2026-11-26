"""Enforce flat, credential-free package boundaries.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

__all__ = ()

PACKAGE = Path(__file__).resolve().parents[1] / "agent_k"


def test_package_is_flat_and_documents_public_boundaries() -> None:
    for path in PACKAGE.rglob("*.py"):
        assert path.parent == PACKAGE, path
        assert path.name in {"__init__.py", "__main__.py"} or re.fullmatch(r"[a-z]+\.py", path.name)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        documentation = ast.get_docstring(tree)
        assert documentation and "MIT License" in documentation, path
        assert isinstance(tree.body[1], ast.ImportFrom), path
        assert tree.body[1].module == "__future__", path
        assert tree.body[1].names[0].asname == "_annotations", path
        assert any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(name, ast.Name) and name.id == "__all__"
                for name in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            for node in tree.body
        ), path


def test_all_modules_import_without_provider_or_storage_credentials() -> None:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not any(
            term in name
            for term in (
                "API_KEY",
                "KAGGLE_",
                "LOGFIRE_",
                "DATABASE_URL",
                "AGENT_K_",
                "AWS_",
            )
        )
    }
    modules = sorted(path.stem for path in PACKAGE.glob("*.py") if path.stem != "__main__")
    program = (
        "import importlib; "
        f"[importlib.import_module('agent_k.' + name) for name in {modules!r}]; "
        "from agent_k.lobbyist import lobbyist_agent; "
        "from agent_k.scientist import scientist_agent; "
        "assert lobbyist_agent.model is None and scientist_agent.model is None"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_imports_are_removed_without_shims() -> None:
    retired = {"adapters", "agents", "core", "embeddings", "infra", "toolsets", "ui"}
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] == "agent_k" and len(parts) > 1:
                    assert parts[1] not in retired, (path, node.lineno)
    assert not (PACKAGE / "evolution" / "framework.py").exists()
    assert not (PACKAGE / "agents" / "evolver.py").exists()


def test_default_cli_version_contract() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "agent_k"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout == "AGENT-K version 0.1.0\n"
