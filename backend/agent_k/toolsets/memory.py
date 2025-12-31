"""Memory tool helpers for AGENT-K agents.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party (alphabetical)
import logfire
from pydantic_ai import RunContext, ToolDefinition  # noqa: TC002
from pydantic_ai.builtin_tools import MemoryTool

_anthropic_memory: Any | None
try:  # pragma: no cover - optional dependency
    from anthropic.lib.tools import _beta_builtin_memory_tool as _anthropic_memory
except ImportError:  # pragma: no cover - optional dependency
    _anthropic_memory = None

if TYPE_CHECKING:
    from anthropic.lib.tools._beta_builtin_memory_tool import BetaAbstractMemoryTool as _MemoryBase
    from pydantic_ai import Agent

elif _anthropic_memory is not None:
    _MemoryBase = _anthropic_memory.BetaAbstractMemoryTool

else:

    class _MemoryBase:  # pragma: no cover - optional dependency
        """Fallback base when Anthropic memory tool is unavailable."""

        pass


# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "AgentKMemoryTool",
    "create_memory_backend",
    "prepare_memory_tool",
    "register_memory_tool",
)

# =============================================================================
# Section 3: Constants
# =============================================================================
_DEFAULT_MEMORY_DIR = Path(os.getenv("AGENT_K_MEMORY_DIR", ".agent_k_memory"))


# =============================================================================
# Section 11: Classes
# =============================================================================
class AgentKMemoryTool(_MemoryBase):  # pragma: no cover - optional dependency
    """File-backed memory implementation for Anthropic MemoryTool."""

    def __init__(self, base_path: Path | None = None) -> None:
        if _anthropic_memory is None:
            raise RuntimeError("anthropic is required to use AgentKMemoryTool")
        super().__init__()
        self._base_path = (base_path or _DEFAULT_MEMORY_DIR).expanduser().resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)

    def view(self, command: Any) -> str:
        """View file contents or list directory entries."""
        with logfire.span("memory.view", path=command.path):
            try:
                path = self._resolve_path(command.path)
            except ValueError as exc:
                return f"Error: {exc}"

            if not path.exists():
                return f"Error: {command.path} not found."

            if path.is_dir():
                entries = []
                for child in sorted(path.iterdir(), key=lambda p: p.name):
                    suffix = "/" if child.is_dir() else ""
                    entries.append(f"{child.name}{suffix}")
                return "\n".join(entries) if entries else "(empty directory)"

            text = self._read_text(path)
            if command.view_range:
                lines = text.splitlines()
                start, end = _normalize_view_range(command.view_range, len(lines))
                return "\n".join(lines[start - 1 : end])
            return text

    def create(self, command: Any) -> str:
        """Create a file with the provided contents."""
        with logfire.span("memory.create", path=command.path):
            try:
                path = self._resolve_path(command.path)
            except ValueError as exc:
                return f"Error: {exc}"

            if path.exists():
                return f"Error: {command.path} already exists."

            self._write_text(path, command.file_text)
            return f"Created {command.path}."

    def str_replace(self, command: Any) -> str:
        """Replace matching text in a file."""
        with logfire.span("memory.str_replace", path=command.path):
            try:
                path = self._resolve_path(command.path)
            except ValueError as exc:
                return f"Error: {exc}"

            if not path.exists():
                return f"Error: {command.path} not found."

            text = self._read_text(path)
            occurrences = text.count(command.old_str)
            if occurrences == 0:
                return f'Error: "{command.old_str}" not found in {command.path}.'

            updated = text.replace(command.old_str, command.new_str)
            self._write_text(path, updated)
            return f"Replaced {occurrences} occurrence(s) in {command.path}."

    def insert(self, command: Any) -> str:
        """Insert text at a specified line in a file."""
        with logfire.span("memory.insert", path=command.path):
            try:
                path = self._resolve_path(command.path)
            except ValueError as exc:
                return f"Error: {exc}"

            if not path.exists():
                return f"Error: {command.path} not found."

            text = self._read_text(path)
            lines = text.splitlines()
            index = max(command.insert_line - 1, 0)
            if index > len(lines):
                index = len(lines)
            lines.insert(index, command.insert_text)
            updated = "\n".join(lines)
            if text.endswith("\n"):
                updated += "\n"
            self._write_text(path, updated)
            return f"Inserted text at line {command.insert_line} in {command.path}."

    def delete(self, command: Any) -> str:
        """Delete a file or directory."""
        with logfire.span("memory.delete", path=command.path):
            try:
                path = self._resolve_path(command.path)
            except ValueError as exc:
                return f"Error: {exc}"

            if not path.exists():
                return f"Error: {command.path} not found."

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return f"Deleted {command.path}."

    def rename(self, command: Any) -> str:
        """Rename a file or directory."""
        with logfire.span("memory.rename", path=command.old_path, new_path=command.new_path):
            try:
                old_path = self._resolve_path(command.old_path)
                new_path = self._resolve_path(command.new_path)
            except ValueError as exc:
                return f"Error: {exc}"

            if not old_path.exists():
                return f"Error: {command.old_path} not found."

            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)
            return f"Renamed {command.old_path} to {command.new_path}."

    def _resolve_path(self, path: str) -> Path:
        candidate = (self._base_path / path).resolve()
        if not candidate.is_relative_to(self._base_path):
            raise ValueError(f"Path escapes memory root: {path}")
        return candidate

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


# =============================================================================
# Section 12: Functions
# =============================================================================


def create_memory_backend(storage_path: Path | None = None) -> AgentKMemoryTool:
    """Create an Anthropic-compatible memory backend.

    Args:
        storage_path: Base directory for memory files.

    Returns:
        Configured AgentKMemoryTool instance.
    """
    return AgentKMemoryTool(base_path=storage_path)


async def prepare_memory_tool(ctx: RunContext[Any]) -> MemoryTool | None:
    """Dynamically enable MemoryTool only for supported providers."""
    if ctx.model.system != "anthropic":
        return None
    return MemoryTool()


def register_memory_tool(
    agent: Agent[Any, Any],
    memory_backend: AgentKMemoryTool,
) -> None:
    """Register the Anthropic MemoryTool handler on an agent.

    Args:
        agent: Agent instance to register the tool on.
        memory_backend: Memory backend implementation.
    """

    @agent.tool_plain(name="memory", prepare=_prepare_memory_definition)
    def memory(**command: Any) -> Any:
        return memory_backend.call(command)


async def _prepare_memory_definition(
    ctx: RunContext[Any],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    if ctx.model.system != "anthropic":
        return None
    return tool_def


def _normalize_view_range(view_range: list[int], total_lines: int) -> tuple[int, int]:
    if not view_range:
        return (1, total_lines)

    if len(view_range) == 1:
        start = end = view_range[0]
    else:
        start, end = view_range[0], view_range[1]

    if start < 1:
        start = 1
    if end < start:
        end = start
    if end > total_lines:
        end = total_lines

    return (start, end)
