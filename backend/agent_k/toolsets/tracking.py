"""Experiment tracking toolset for AGENT-K.

@notice: |
    Experiment tracking toolset for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.toolsets.tracking
    provides:
        - agent_k.toolsets.tracking:tracking_toolset
        - agent_k.toolsets.tracking:tracking_record_experiment
        - agent_k.toolsets.tracking:tracking_list_experiments
        - agent_k.toolsets.tracking:tracking_best_experiment
    pattern: toolset

@similar:
    - id: agent_k.core.tracking
        when: "Core tracking storage APIs; this module exposes tool wrappers."

@agent-guidance:
    do:
        - "Use agent_k.toolsets.tracking as the canonical home for this capability."
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

import asyncio
from functools import cache
from typing import Annotated, Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from agent_k.core.sage import Doc, Range
from agent_k.core.tracking import ExperimentRecord, ExperimentTracker, create_experiment_tracker

__all__ = ("tracking_toolset",)

tracking_toolset: FunctionToolset[Any] = FunctionToolset(id="tracking")


@cache
def _tracker() -> ExperimentTracker:
    """Return the process-wide experiment tracker.

    @notice: |
        Reuses one tracker so schema setup is not repeated per tool call.

    @dev: |
        Constructing a tracker re-runs schema DDL and opens a connection. Tools are called
        many times per mission, so the instance is cached for the process lifetime.
    """
    return create_experiment_tracker()


@tracking_toolset.tool
async def tracking_record_experiment(
    ctx: RunContext[Any], record: Annotated[dict[str, Any], Doc("Experiment record payload.")]
) -> dict[str, Any]:
    """Record an experiment to the persistent tracker.

    @notice: |
        Validates and stores the experiment record.
    """
    _ = ctx
    tracker = _tracker()
    entry = ExperimentRecord.model_validate(record)
    stored = await asyncio.to_thread(tracker.record_experiment, entry)
    return tracker.summarize(stored).model_dump(mode="json")


@tracking_toolset.tool
async def tracking_list_experiments(
    ctx: RunContext[Any],
    competition_id: Annotated[str, Doc("Competition identifier (slug).")],
    limit: Annotated[int, Doc("Maximum experiments to return."), Range(1, 500)] = 25,
) -> list[dict[str, Any]]:
    """List recent experiments for a competition.

    @notice: |
        Returns summarized experiment records.
    """
    _ = ctx
    tracker = _tracker()
    records = await asyncio.to_thread(tracker.list_experiments, competition_id, limit=limit)
    return [tracker.summarize(record).model_dump(mode="json") for record in records]


@tracking_toolset.tool
async def tracking_best_experiment(
    ctx: RunContext[Any],
    competition_id: Annotated[str, Doc("Competition identifier (slug).")],
    metric: Annotated[str, Doc("Metric name to rank by.")] = "public_score",
    direction: Annotated[str, Doc("Rank direction: maximize or minimize.")] = "maximize",
) -> dict[str, Any] | None:
    """Return the best experiment for a competition and metric.

    @notice: |
        Returns the top experiment summary or None.
    """
    _ = ctx
    tracker = _tracker()
    record = await asyncio.to_thread(tracker.best_experiment, competition_id, metric=metric, direction=direction)
    return tracker.summarize(record).model_dump(mode="json") if record else None
