"""Tests for the memory tool helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pytest

from agent_k.toolsets.memory import create_memory_backend

__all__ = ()

pytest.importorskip("anthropic")

if TYPE_CHECKING:
    from pathlib import Path


def test_create_and_view(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    created = backend.call(
        {
            "command": "create",
            "path": "shared/target_competition.md",
            "file_text": "Titanic competition details",
        }
    )
    assert "Created" in created

    viewed = backend.call(
        {
            "command": "view",
            "path": "shared/target_competition.md",
        }
    )
    assert "Titanic competition details" in viewed


def test_str_replace(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    backend.call(
        {
            "command": "create",
            "path": "shared/notes.txt",
            "file_text": "alpha beta",
        }
    )

    replaced = backend.call(
        {
            "command": "str_replace",
            "path": "shared/notes.txt",
            "old_str": "beta",
            "new_str": "gamma",
        }
    )
    assert "Replaced" in replaced

    viewed = backend.call(
        {
            "command": "view",
            "path": "shared/notes.txt",
        }
    )
    assert "alpha gamma" in viewed
