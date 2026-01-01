"""Kaggle toolset for AGENT-K agents.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

# Third-party (alphabetical)
import logfire
from pydantic_ai import RunContext  # noqa: TC002
from pydantic_ai.toolsets import FunctionToolset

# Local imports (core first, then alphabetical)
from agent_k.core.deps import KaggleDeps

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_k.core.models import Competition
    from agent_k.core.protocols import PlatformAdapter

P = ParamSpec("P")
ToolResultT = TypeVar("ToolResultT")

__all__ = ("KaggleDeps", "kaggle_toolset")

# =============================================================================
# Toolset Definition
# =============================================================================
kaggle_toolset: FunctionToolset[Any] = FunctionToolset(id="kaggle")

# Cache for competition data
_cache: dict[str, Competition] = {}


# =============================================================================
# Tool Helpers
# =============================================================================
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

    def decorator(
        func: Callable[P, Awaitable[ToolResultT]],
    ) -> Callable[P, Awaitable[ToolResultT]]:
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


# =============================================================================
# Tool Implementations
# =============================================================================
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
    categories: list[str] | None = None,
    keywords: list[str] | None = None,
    min_prize: int | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """Search Kaggle for active competitions."""
    with logfire.span("kaggle_search_competitions", categories=categories, keywords=keywords):
        adapter = _require_adapter(ctx)

        competitions: list[dict[str, Any]] = []

        async for comp in adapter.search_competitions(
            categories=categories,
            keywords=keywords,
            min_prize=min_prize,
            active_only=active_only,
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
    ctx: RunContext[Any],
    competition_id: str,
) -> dict[str, Any]:
    """Get detailed information about a specific Kaggle competition."""
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
    competition_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Get the current leaderboard for a competition."""
    with logfire.span("kaggle_get_leaderboard", competition_id=competition_id):
        adapter = _require_adapter(ctx)
        entries = await adapter.get_leaderboard(competition_id, limit=limit)
        return {
            "competition_id": competition_id,
            "total_entries": len(entries),
            "entries": [
                {
                    "rank": e.rank,
                    "team_name": e.team_name,
                    "score": e.score,
                }
                for e in entries
            ],
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
    ctx: RunContext[Any],
    competition_id: str,
) -> dict[str, Any]:
    """List available datasets for a competition."""
    with logfire.span("kaggle_list_datasets", competition_id=competition_id):
        adapter = _require_adapter(ctx)
        request = getattr(adapter, "_request", None)
        if request is None:
            raise RuntimeError("Adapter does not support listing datasets")

        response = await request(
            "GET",
            f"/competitions/data/list/{competition_id}",
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list datasets: {response.status_code}")

        files = response.json()
        return {
            "competition_id": competition_id,
            "files": [
                {
                    "name": f.get("name"),
                    "size": f.get("totalBytes"),
                    "description": f.get("description"),
                }
                for f in files
            ],
        }


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
        await handler(
            task_id=task_id,
            tool_call_id=tool_call_id,
            tool_type=tool_type,
            operation=operation,
        )
        return
    if method == "emit_tool_error":
        await handler(
            task_id=task_id,
            tool_call_id=tool_call_id,
            error=error or "Unknown error",
        )
        return
    await handler(
        task_id=task_id,
        tool_call_id=tool_call_id,
        result=result,
        duration_ms=duration_ms or 0,
    )
