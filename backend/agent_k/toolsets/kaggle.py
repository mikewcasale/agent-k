"""Kaggle toolset for AGENT-K agents.

@notice: |
    Kaggle toolset for AGENT-K agents.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.toolsets.kaggle
    provides:
        - agent_k.toolsets.kaggle:kaggle_toolset
        - agent_k.toolsets.kaggle:kaggle_search_competitions
        - agent_k.toolsets.kaggle:kaggle_get_competition
        - agent_k.toolsets.kaggle:kaggle_get_leaderboard
        - agent_k.toolsets.kaggle:kaggle_list_datasets
    pattern: toolset

@similar:
    - id: agent_k.adapters.kaggle
        when: "Adapter implementation backing these tools."

@agent-guidance:
    do:
        - "Use agent_k.toolsets.kaggle as the canonical home for this capability."
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

import time
from functools import wraps
from typing import TYPE_CHECKING, Annotated, Any, ParamSpec, TypeVar, cast

import logfire
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from agent_k.core.deps import KaggleDeps
from agent_k.core.sage import Doc, Range

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_k.core.models import Competition
    from agent_k.core.protocols import PlatformAdapter

P = ParamSpec("P")
"""Parameter specification for tool wrapper callables."""

ToolResultT = TypeVar("ToolResultT")
"""Type variable for tool result payloads."""

__all__ = ("KaggleDeps", "kaggle_toolset")

kaggle_toolset: FunctionToolset[Any] = FunctionToolset(id="kaggle")

# Cache for competition data
_cache: dict[str, Competition] = {}


def _error_dict_response(error: str) -> dict[str, Any]:
    return {"error": error}


def _error_list_response(error: str) -> list[dict[str, Any]]:
    return [{"error": error}]


def _search_summary(result: list[dict[str, Any]]) -> dict[str, Any]:
    return {"count": len(result)}


def _competition_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {"id": result.get("id")}


def _leaderboard_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {"total_entries": result.get("total_entries", 0)}


def _dataset_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {"file_count": len(result.get("files", []))}


def with_tool_telemetry(
    *,
    task_id: str,
    tool_type: str,
    operation: str,
    error_response: Callable[[str], ToolResultT],
    result_summary: Callable[[ToolResultT], dict[str, Any]],
) -> Callable[[Callable[P, Awaitable[ToolResultT]]], Callable[P, Awaitable[ToolResultT]]]:
    """Wrap a tool function with standard telemetry and error handling."""

    def decorator(func: Callable[P, Awaitable[ToolResultT]]) -> Callable[P, Awaitable[ToolResultT]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResultT:
            ctx_obj = args[0] if args else kwargs.get("ctx")
            if ctx_obj is None:
                raise RuntimeError("RunContext is required for tool telemetry")
            ctx = cast("RunContext[Any]", ctx_obj)
            tool_call_id = f"{task_id}_{id(ctx):x}"
            start_time = time.perf_counter()
            await _emit_tool_event(
                ctx,
                "emit_tool_start",
                task_id=task_id,
                tool_call_id=tool_call_id,
                tool_type=tool_type,
                operation=operation,
            )

            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                await _emit_tool_event(
                    ctx,
                    "emit_tool_error",
                    task_id=task_id,
                    tool_call_id=tool_call_id,
                    tool_type=tool_type,
                    operation=operation,
                    error=str(exc),
                )
                return error_response(str(exc))

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            await _emit_tool_event(
                ctx,
                "emit_tool_result",
                task_id=task_id,
                tool_call_id=tool_call_id,
                tool_type=tool_type,
                operation=operation,
                result=result_summary(result),
                duration_ms=duration_ms,
            )
            return result

        return wrapper

    return decorator


def _require_adapter(ctx: RunContext[Any]) -> PlatformAdapter:
    adapter = _resolve_adapter(ctx)
    if adapter is None:
        raise RuntimeError("Kaggle adapter is not configured")
    return adapter


def _serialize_competition(comp: Competition) -> dict[str, Any]:
    return {
        "id": comp.id,
        "title": comp.title,
        "description": comp.description[:500] if comp.description else None,
        "type": comp.competition_type.value,
        "metric": comp.metric.value,
        "metric_direction": comp.metric_direction,
        "days_remaining": comp.days_remaining,
        "deadline": comp.deadline.isoformat(),
        "prize_pool": comp.prize_pool,
        "max_team_size": comp.max_team_size,
        "max_daily_submissions": comp.max_daily_submissions,
        "tags": list(comp.tags) if comp.tags else [],
    }


@kaggle_toolset.tool
@with_tool_telemetry(
    task_id="kaggle_search",
    tool_type="kaggle_mcp",
    operation="competitions.list",
    error_response=_error_list_response,
    result_summary=_search_summary,
)
async def kaggle_search_competitions(
    ctx: RunContext[Any],
    categories: Annotated[list[str] | None, Doc("Competition categories to filter.")] = None,
    keywords: Annotated[list[str] | None, Doc("Keyword filters for search.")] = None,
    min_prize: Annotated[int | None, Doc("Minimum prize pool in USD."), Range(0, 1_000_000_000)] = None,
    active_only: Annotated[bool, Doc("Only return active competitions.")] = True,
) -> list[dict[str, Any]]:
    """Search Kaggle for active competitions.

    @notice: |
        Returns metadata for competitions matching search criteria.

    @effects:
        io:
            - Kaggle API request
        state:
            - in-module cache
    """
    with logfire.span("kaggle_search_competitions", categories=categories, keywords=keywords):
        adapter = _require_adapter(ctx)

        competitions: list[dict[str, Any]] = []

        async for comp in adapter.search_competitions(
            categories=categories, keywords=keywords, min_prize=min_prize, active_only=active_only
        ):
            _store_competition(ctx, comp)
            competitions.append(
                {
                    "id": comp.id,
                    "title": comp.title,
                    "type": comp.competition_type.value,
                    "metric": comp.metric.value,
                    "days_remaining": comp.days_remaining,
                    "prize_pool": comp.prize_pool,
                    "tags": list(comp.tags) if comp.tags else [],
                    "is_active": comp.is_active,
                }
            )
            max_results = getattr(ctx.deps, "max_results", 50) or 50
            if len(competitions) >= max_results:
                break

        return competitions


@kaggle_toolset.tool
@with_tool_telemetry(
    task_id="kaggle_competition",
    tool_type="kaggle_mcp",
    operation="competitions.get",
    error_response=_error_dict_response,
    result_summary=_competition_summary,
)
async def kaggle_get_competition(
    ctx: RunContext[Any], competition_id: Annotated[str, Doc("Competition identifier (slug).")]
) -> dict[str, Any]:
    """Get detailed information about a specific Kaggle competition.

    @notice: |
        Fetches competition metadata and caches results.
    """
    with logfire.span("kaggle_get_competition", competition_id=competition_id):
        adapter = _require_adapter(ctx)

        if competition_id in _cache:
            comp = _cache[competition_id]
        else:
            comp = await adapter.get_competition(competition_id)
            _store_competition(ctx, comp)

        return _serialize_competition(comp)


@kaggle_toolset.tool
@with_tool_telemetry(
    task_id="kaggle_leaderboard",
    tool_type="kaggle_mcp",
    operation="competitions.leaderboard",
    error_response=_error_dict_response,
    result_summary=_leaderboard_summary,
)
async def kaggle_get_leaderboard(
    ctx: RunContext[Any],
    competition_id: Annotated[str, Doc("Competition identifier (slug).")],
    limit: Annotated[int, Doc("Maximum entries to return."), Range(1, 1000)] = 20,
) -> dict[str, Any]:
    """Get the current leaderboard for a competition.

    @notice: |
        Returns leaderboard entries ordered by rank.
    """
    with logfire.span("kaggle_get_leaderboard", competition_id=competition_id):
        adapter = _require_adapter(ctx)
        entries = await adapter.get_leaderboard(competition_id, limit=limit)
        return {
            "competition_id": competition_id,
            "total_entries": len(entries),
            "entries": [{"rank": e.rank, "team_name": e.team_name, "score": e.score} for e in entries],
        }


@kaggle_toolset.tool
@with_tool_telemetry(
    task_id="kaggle_datasets",
    tool_type="kaggle_mcp",
    operation="competitions.data",
    error_response=_error_dict_response,
    result_summary=_dataset_summary,
)
async def kaggle_list_datasets(
    ctx: RunContext[Any], competition_id: Annotated[str, Doc("Competition identifier (slug).")]
) -> dict[str, Any]:
    """List available datasets for a competition.

    @notice: |
        Returns file metadata for competition datasets.
    """
    with logfire.span("kaggle_list_datasets", competition_id=competition_id):
        adapter = _require_adapter(ctx)
        request = getattr(adapter, "_request", None)
        if request is None:
            raise RuntimeError("Adapter does not support listing datasets")

        response = await request("GET", f"/competitions/data/list/{competition_id}")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list datasets: {response.status_code}")

        return {"competition_id": competition_id, "files": _parse_dataset_files(response.json())}


def _parse_dataset_files(payload: Any) -> list[dict[str, Any]]:
    """Normalize Kaggle ``/competitions/data/list`` payloads to file records.

    Kaggle historically returned a bare list of file entries, but current
    responses wrap the list under a ``files`` key. Individual entries may
    be plain filenames (str), full metadata dicts with ``name``/``totalBytes``,
    or the ``nameNullable`` variant used by newer API versions.
    """
    raw_entries: Any
    if isinstance(payload, dict):
        raw_entries = payload.get("files", [])
    else:
        raw_entries = payload
    if not isinstance(raw_entries, list):
        return []

    files: list[dict[str, Any]] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            files.append({"name": entry, "size": None, "description": None})
            continue
        if not isinstance(entry, dict):
            continue
        files.append(
            {
                "name": entry.get("name") or entry.get("nameNullable"),
                "size": entry.get("totalBytes") if entry.get("totalBytes") is not None else entry.get("size"),
                "description": entry.get("description"),
            }
        )
    return files


def _resolve_adapter(ctx: RunContext[Any]) -> PlatformAdapter | None:
    adapter = getattr(ctx.deps, "kaggle_adapter", None)
    if adapter is None:
        adapter = getattr(ctx.deps, "platform_adapter", None)
    return adapter


def _store_competition(ctx: RunContext[Any], competition: Competition) -> None:
    _cache[competition.id] = competition
    search_cache = getattr(ctx.deps, "search_cache", None)
    if isinstance(search_cache, dict):
        search_cache[competition.id] = competition


async def _emit_tool_event(
    ctx: RunContext[Any],
    method: str,
    *,
    task_id: str,
    tool_call_id: str,
    tool_type: str,
    operation: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    duration_ms: int | None = None,
) -> None:
    emitter = getattr(ctx.deps, "event_emitter", None)
    if emitter is None:
        return
    handler = getattr(emitter, method, None)
    if handler is None:
        return
    if method == "emit_tool_start":
        await handler(task_id=task_id, tool_call_id=tool_call_id, tool_type=tool_type, operation=operation)
        return
    if method == "emit_tool_error":
        await handler(task_id=task_id, tool_call_id=tool_call_id, error=error or "Unknown error")
        return
    await handler(task_id=task_id, tool_call_id=tool_call_id, result=result, duration_ms=duration_ms or 0)
