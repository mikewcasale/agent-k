"""Experiment tracking toolset for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from typing import Any

# Third-party (alphabetical)
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

# Local imports (core first, then alphabetical)
from agent_k.core.tracking import ExperimentRecord, ExperimentTracker

__all__ = ("tracking_toolset",)

tracking_toolset: FunctionToolset[Any] = FunctionToolset(id="tracking")


@tracking_toolset.tool
async def tracking_record_experiment(ctx: RunContext[Any], record: dict[str, Any]) -> dict[str, Any]:
    """Record an experiment to the persistent tracker."""
    _ = ctx
    tracker = ExperimentTracker()
    entry = ExperimentRecord.model_validate(record)
    stored = tracker.record_experiment(entry)
    return tracker.summarize(stored).model_dump(mode="json")


@tracking_toolset.tool
async def tracking_list_experiments(ctx: RunContext[Any], competition_id: str, limit: int = 25) -> list[dict[str, Any]]:
    """List recent experiments for a competition."""
    _ = ctx
    tracker = ExperimentTracker()
    records = tracker.list_experiments(competition_id, limit=limit)
    return [tracker.summarize(record).model_dump(mode="json") for record in records]


@tracking_toolset.tool
async def tracking_best_experiment(
    ctx: RunContext[Any], competition_id: str, metric: str = "public_score", direction: str = "maximize"
) -> dict[str, Any] | None:
    """Return the best experiment for a competition and metric."""
    _ = ctx
    tracker = ExperimentTracker()
    record = tracker.best_experiment(competition_id, metric=metric, direction=direction)
    return tracker.summarize(record).model_dump(mode="json") if record else None
