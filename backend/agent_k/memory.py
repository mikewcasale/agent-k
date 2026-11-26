"""Public Pydantic AI tools for mission-scoped, durable memory records.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

__all__ = ("MemoryDeps", "memory_toolset", "read_memory", "write_memory")


@dataclass
class MemoryDeps:
    """Directory owned by the current mission process."""

    memory_root: Path


memory_toolset: FunctionToolset[MemoryDeps] = FunctionToolset(id="memory")


@memory_toolset.tool
def read_memory(ctx: RunContext[MemoryDeps], name: str) -> str:
    """Read an existing mission note.

    Args:
        ctx: Current mission dependencies.
        name: A simple note identifier using letters, numbers, or hyphens.
    """
    return _note_path(ctx.deps.memory_root, name).read_text(encoding="utf-8")


@memory_toolset.tool
def write_memory(ctx: RunContext[MemoryDeps], name: str, content: str) -> str:
    """Atomically write a bounded mission note.

    Args:
        ctx: Current mission dependencies.
        name: A simple note identifier using letters, numbers, or hyphens.
        content: Evidence or reasoning to retain for this mission.
    """
    if len(content.encode("utf-8")) > 65536:
        raise ValueError("Memory notes are limited to 64 KiB")
    path = _note_path(ctx.deps.memory_root, name)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return name


def _note_path(root: Path, name: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9-]{1,100}", name):
        raise ValueError("Invalid memory note identifier")
    path = root / f"{name}.md"
    if path.is_symlink():
        raise ValueError("Memory notes may not be symbolic links")
    return path
