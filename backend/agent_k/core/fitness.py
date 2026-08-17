"""Canonical score/fitness conversions for AGENT-K.

@notice: |
    Canonical score/fitness conversions for AGENT-K.

@dev: |
    Every producer and consumer of a fitness value (the OpenEvolve evaluator,
    the Evolver agent, the strategy fitness factory) must use these helpers so
    that a single convention holds end to end: fitness is always non-negative
    and higher-is-better, regardless of metric direction. Failed or invalid
    evaluations report ``FITNESS_FLOOR`` so they can never outrank a solution
    that actually ran.

@graph:
    id: agent_k.core.fitness
    provides:
        - agent_k.core.fitness
    pattern: value-conversion

@similar:
    - id: agent_k.core.strategy
        when: "Building fitness policies and functions rather than converting values."

@agent-guidance:
    do:
        - "Use agent_k.core.fitness as the canonical home for this capability."
        - "Convert scores with score_to_fitness before comparing solutions."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Negate scores to express minimize direction; fitness must stay non-negative."

@human-review:
    last-verified: 2026-08-17
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .types import MetricDirection

__all__ = ("FITNESS_FLOOR", "coerce_metric_direction", "fitness_to_score", "score_to_fitness")

FITNESS_FLOOR: Final[float] = 0.0
"""Fitness reported for invalid, failed, or unscored evaluations."""


def coerce_metric_direction(direction: str) -> MetricDirection:
    """Normalize a free-form direction string to a MetricDirection.

    @notice: |
        Normalizes a direction string, defaulting to "maximize".

    @dev: |
        Competition metadata and OpenEvolve context payloads carry the direction
        as a plain string. Anything that is not exactly "minimize" (case and
        whitespace insensitive) is treated as "maximize".
    """
    return "minimize" if direction.strip().lower() == "minimize" else "maximize"


def score_to_fitness(score: float | None, direction: MetricDirection) -> float:
    """Convert a metric score into non-negative, higher-is-better fitness.

    @notice: |
        Maps a raw metric score onto the canonical AGENT-K fitness scale.

    @dev: |
        For "minimize" metrics the score is mapped through ``1 / (1 + score)``,
        which is strictly decreasing on ``[0, inf)`` and lands in ``(0, 1]`` so a
        valid solution always beats ``FITNESS_FLOOR``. For "maximize" metrics the
        score is passed through, clamped at ``FITNESS_FLOOR``. ``None`` scores
        report ``FITNESS_FLOOR``.
    """
    if score is None:
        return FITNESS_FLOOR
    value = float(score)
    if direction == "minimize":
        return 1.0 / (1.0 + max(value, 0.0))
    return max(value, FITNESS_FLOOR)


def fitness_to_score(fitness: float | None, direction: MetricDirection) -> float | None:
    """Invert :func:`score_to_fitness` back to a metric score.

    @notice: |
        Recovers the metric score behind a canonical fitness value.

    @dev: |
        Returns ``None`` when the fitness carries no recoverable score: a missing
        value, or a non-positive minimize-direction fitness (which encodes a
        failed evaluation rather than a real score). Scores clamped by
        :func:`score_to_fitness` are not recoverable and round-trip to ``0.0``.
    """
    if fitness is None:
        return None
    value = float(fitness)
    if direction == "minimize":
        if value <= FITNESS_FLOOR:
            return None
        return (1.0 / value) - 1.0
    return value
