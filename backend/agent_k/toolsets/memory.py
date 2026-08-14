"""Memory tool helpers for AGENT-K agents.

@notice: |
    Memory tool helpers for AGENT-K agents.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.toolsets.memory
    provides:
        - agent_k.toolsets.memory:AgentKMemoryTool
        - agent_k.toolsets.memory:create_memory_backend
        - agent_k.toolsets.memory:prepare_memory_tool
        - agent_k.toolsets.memory:register_memory_tool
    pattern: toolset

@similar:
    - id: agent_k.embeddings.store
        when: "Vector store persistence; this module is file-backed memory tool."

@agent-guidance:
    do:
        - "Use agent_k.toolsets.memory as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import logfire
from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.builtin_tools import MemoryTool

from agent_k.core.sage import Doc

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


__all__ = ("AgentKMemoryTool", "create_memory_backend", "prepare_memory_tool", "register_memory_tool")

_DEFAULT_MEMORY_DIR = Path(os.getenv("AGENT_K_MEMORY_DIR", ".agent_k_memory"))


class AgentKMemoryTool(_MemoryBase):  # pragma: no cover - optional dependency
    """File-backed memory implementation for Anthropic MemoryTool.

    @notice: |
        File-backed memory implementation for Anthropic MemoryTool.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: memory-tool
            rationale: "Provides a file-backed memory interface for agents."
            violations: "Bypassing this tool breaks memory consistency."

        @concurrency:
            model: asyncio
            safe: false
            reason: "Performs filesystem mutations without locks."
    """

    def __init__(self, base_path: Annotated[Path | None, Doc("Base path for memory storage.")] = None) -> None:
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
                entries = [
                    f"{child.name}{'/' if child.is_dir() else ''}"
                    for child in sorted(path.iterdir(), key=lambda p: p.name)
                ]
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

            if command.file_text is None:
                return "Error: file_text must be a string."

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

            if command.old_str is None or command.new_str is None:
                return "Error: old_str and new_str must be strings."

            text = self._read_text(path)
            occurrences = text.count(command.old_str)
            if occurrences == 0:
                return f'Error: "{command.old_str}" not found in {command.path}.'
            if occurrences > 1:
                return (
                    f'Error: "{command.old_str}" appears {occurrences} times in {command.path}; '
                    "old_str must uniquely identify the text to replace."
                )

            updated = text.replace(command.old_str, command.new_str, 1)
            self._write_text(path, updated)
            return f"Replaced 1 occurrence in {command.path}."

    def insert(self, command: Any) -> str:
        """Insert text at a specified line in a file."""
        with logfire.span("memory.insert", path=command.path):
            try:
                path = self._resolve_path(command.path)
            except ValueError as exc:
                return f"Error: {exc}"

            if not path.exists():
                return f"Error: {command.path} not found."

            if command.insert_text is None:
                return "Error: insert_text must be a string."

            text = self._read_text(path)
            lines = text.splitlines()
            # Anthropic memory tool spec: insert_line is the line number AFTER which to insert
            # text (0 = insert at the beginning of the file, len(lines) = append at end).
            index = max(0, min(int(command.insert_line), len(lines)))
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


def create_memory_backend(
    storage_path: Annotated[Path | None, Doc("Base directory for memory files.")] = None,
) -> AgentKMemoryTool:
    """Create an Anthropic-compatible memory backend.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Creates a file-backed memory tool for Anthropic providers.

        @factory-for:
            id: agent_k.toolsets.memory:AgentKMemoryTool
            rationale: "Centralizes default storage path behavior."
            singleton: false
            cache-key: storage_path

        @canonical-home:
            for:
                - "memory backend construction"
            notes: "Use create_memory_backend to ensure defaults."
    """
    return AgentKMemoryTool(base_path=storage_path)


async def prepare_memory_tool(
    ctx: Annotated[RunContext[Any], Doc("Run context for tool preparation.")],
) -> MemoryTool | None:
    """Dynamically enable MemoryTool only for supported providers.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Returns MemoryTool only for Anthropic models.
    """
    return None if ctx.model.system != "anthropic" else MemoryTool()


def register_memory_tool(
    agent: Annotated[Agent[Any, Any], Doc("Agent instance to register the tool on.")],
    memory_backend: Annotated[AgentKMemoryTool, Doc("Memory backend implementation.")],
) -> None:
    """Register the Anthropic MemoryTool handler on an agent.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Attaches the memory tool to the agent with a plain tool handler.

        @effects:
            state:
                - agent tool registry
    """

    @agent.tool_plain(name="memory", prepare=_prepare_memory_definition)
    def memory(**command: Any) -> Any:
        return memory_backend.call(command)


async def _prepare_memory_definition(ctx: RunContext[Any], tool_def: ToolDefinition) -> ToolDefinition | None:
    return None if ctx.model.system != "anthropic" else tool_def


def _normalize_view_range(view_range: list[int], total_lines: int) -> tuple[int, int]:
    if not view_range:
        return (1, total_lines)

    if len(view_range) == 1:
        start = end = view_range[0]
    else:
        start, end = view_range[0], view_range[1]

    start = max(start, 1)
    end = max(end, start)
    end = min(end, total_lines)
    return (start, end)
