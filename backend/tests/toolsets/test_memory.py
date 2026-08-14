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
        {"command": "create", "path": "shared/target_competition.md", "file_text": "Titanic competition details"}
    )
    assert "Created" in created

    viewed = backend.call({"command": "view", "path": "shared/target_competition.md"})
    assert "Titanic competition details" in viewed


def test_str_replace(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    backend.call({"command": "create", "path": "shared/notes.txt", "file_text": "alpha beta"})

    replaced = backend.call(
        {"command": "str_replace", "path": "shared/notes.txt", "old_str": "beta", "new_str": "gamma"}
    )
    assert "Replaced" in replaced

    viewed = backend.call({"command": "view", "path": "shared/notes.txt"})
    assert "alpha gamma" in viewed


def test_str_replace_rejects_non_unique_match(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    backend.call({"command": "create", "path": "notes.txt", "file_text": "foo bar foo baz foo"})

    result = backend.call({"command": "str_replace", "path": "notes.txt", "old_str": "foo", "new_str": "qux"})

    assert isinstance(result, str)
    assert result.startswith("Error:")
    assert "3 times" in result

    viewed = backend.call({"command": "view", "path": "notes.txt"})
    assert "foo bar foo baz foo" in viewed


def test_str_replace_reports_not_found(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    backend.call({"command": "create", "path": "notes.txt", "file_text": "alpha"})

    result = backend.call({"command": "str_replace", "path": "notes.txt", "old_str": "missing", "new_str": "gamma"})

    assert isinstance(result, str)
    assert result.startswith("Error:")
    assert "not found" in result


def test_insert_places_text_after_target_line(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    backend.call({"command": "create", "path": "notes.txt", "file_text": "line1\nline2\nline3\n"})

    backend.call({"command": "insert", "path": "notes.txt", "insert_line": 1, "insert_text": "inserted"})

    viewed = backend.call({"command": "view", "path": "notes.txt"})
    assert isinstance(viewed, str)
    assert viewed.splitlines() == ["line1", "inserted", "line2", "line3"]


def test_insert_at_beginning_when_insert_line_zero(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    backend.call({"command": "create", "path": "notes.txt", "file_text": "line1\nline2\n"})

    backend.call({"command": "insert", "path": "notes.txt", "insert_line": 0, "insert_text": "top"})

    viewed = backend.call({"command": "view", "path": "notes.txt"})
    assert isinstance(viewed, str)
    assert viewed.splitlines() == ["top", "line1", "line2"]


def test_insert_at_end_when_insert_line_beyond_length(tmp_path: Path) -> None:
    backend = create_memory_backend(tmp_path)

    backend.call({"command": "create", "path": "notes.txt", "file_text": "line1\nline2"})

    backend.call({"command": "insert", "path": "notes.txt", "insert_line": 999, "insert_text": "tail"})

    viewed = backend.call({"command": "view", "path": "notes.txt"})
    assert isinstance(viewed, str)
    assert viewed.splitlines() == ["line1", "line2", "tail"]
