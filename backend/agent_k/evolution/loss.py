"""Loss function evolution for LightGBM objectives.

@notice: |
    Loss function evolution for LightGBM objectives.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evolution.loss
    provides:
        - agent_k.evolution.loss
    pattern: loss-functions

@agent-guidance:
    do:
        - "Use agent_k.evolution.loss as the canonical home for this capability."
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

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from agent_k.evolution.framework import FitnessFn, Individual, Population

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

type LossObjective = str
"""LightGBM objective identifiers."""

type LightGBMObjective = Callable[..., tuple[NDArray[np.float64], NDArray[np.float64]]]
"""Callable accepted by LightGBM as a custom objective; returns (gradient, hessian)."""

__all__ = (
    "LOSS_OBJECTIVES",
    "LightGBMObjective",
    "LossFunctionEvolver",
    "LossGenome",
    "build_custom_objective",
    "build_lightgbm_objective_params",
    "render_lightgbm_objective_source",
)

LOSS_OBJECTIVES: Final[tuple[LossObjective, ...]] = ("regression", "regression_l1", "huber", "quantile")
"""Regression objectives the loss evolver searches over."""

_L1_LIKE_OBJECTIVES: Final[frozenset[str]] = frozenset({"regression_l1", "huber"})
_FLOAT_PRECISION: Final[int] = 12
_MIN_DELTA: Final[float] = 1e-3
_MIN_HESSIAN: Final[float] = 1e-6
_MIN_QUANTILE_ALPHA: Final[float] = 1e-3
_MIN_WEIGHT: Final[float] = 1e-3
_OBJECTIVE_SOURCE_HEADER: Final[str] = '''\
def custom_objective(first, second):
    """Evolved LightGBM objective ({objective}); returns (grad, hess).

    Works with both LightGBM APIs: lgb.train passes (preds, Dataset) while the
    sklearn wrapper passes (y_true, y_pred).
    """
    if hasattr(second, "get_label"):
        y_pred = np.asarray(first, dtype=float)
        y_true = np.asarray(second.get_label(), dtype=float)
    else:
        y_true = np.asarray(first, dtype=float)
        y_pred = np.asarray(second, dtype=float)
    residual = y_pred - y_true
    scaled = residual / {delta}
    denom = np.sqrt(1.0 + scaled * scaled)
    pseudo_grad = residual / denom
    pseudo_hess = 1.0 / (denom * denom * denom)
'''
_QUANTILE_SOURCE_BODY: Final[str] = """\
    side = np.where(residual < 0.0, {alpha}, {one_minus_alpha})
    grad = side * pseudo_grad / {delta}
    hess = side * pseudo_hess / {delta}
"""
_BLEND_SOURCE_BODY: Final[str] = """\
    grad = {inverse_blend} * residual + {blend} * pseudo_grad
    hess = {inverse_blend} + {blend} * pseudo_hess
"""
_OBJECTIVE_SOURCE_FOOTER: Final[str] = """\
    asymmetry = np.where(residual > 0.0, {weight}, 1.0)
    return grad * asymmetry, np.maximum(hess * asymmetry, {min_hessian})
"""


@dataclass(slots=True)
class LossGenome:
    """Genome for evolving custom objective functions.

    @notice: |
        Genome for evolving custom objective functions.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: genome-model
            rationale: "Dataclass for LightGBM loss function parameters."
    """

    objective: LossObjective = "regression"
    asymmetric_weight: float = 1.0
    huber_delta: float = 1.0
    mae_rmse_blend: float = 0.5
    quantile_alpha: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class LossFunctionEvolver:
    """Evolve LightGBM loss function parameters via genetic search.

    @notice: |
        Evolve LightGBM loss function parameters via genetic search.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: evolver
            rationale: "Coordinates evolutionary search over loss function genomes."
    """

    _objectives: tuple[LossObjective, ...] = LOSS_OBJECTIVES

    def __init__(
        self, fitness_fn: FitnessFn[LossGenome], *, population_size: int = 16, rng: random.Random | None = None
    ) -> None:
        self._fitness_fn = fitness_fn
        self._population_size = population_size
        self._rng = rng or random.Random()
        self._population = self._initialize_population()

    def evolve(self, *, generations: int = 5) -> dict[str, Any]:
        """Run evolution for a fixed number of generations."""
        for _ in range(generations):
            self._population.evolve_generation(
                fitness_fn=self._fitness_fn, mutation_fn=self._mutate, crossover_fn=self._crossover, mutation_rate=0.3
            )
        best = self._population.best()
        return {
            "best_genome": best.genome if best else None,
            "best_fitness": best.fitness if best else None,
            "population": self._population,
        }

    def _initialize_population(self) -> Population[LossGenome]:
        individuals = [Individual(self._random_genome()) for _ in range(self._population_size)]
        return Population(individuals, rng=self._rng)

    def _random_genome(self) -> LossGenome:
        return LossGenome(
            objective=self._rng.choice(self._objectives),
            asymmetric_weight=self._rng.uniform(0.5, 2.0),
            huber_delta=self._rng.uniform(0.5, 5.0),
            mae_rmse_blend=self._rng.uniform(0.0, 1.0),
            quantile_alpha=self._rng.uniform(0.05, 0.95),
        )

    def _mutate(self, genome: LossGenome, rng: random.Random) -> LossGenome:
        return LossGenome(
            objective=rng.choice(self._objectives) if rng.random() < 0.2 else genome.objective,
            asymmetric_weight=_clip(genome.asymmetric_weight + rng.uniform(-0.2, 0.2), 0.1, 3.0),
            huber_delta=_clip(genome.huber_delta + rng.uniform(-0.3, 0.3), 0.1, 10.0),
            mae_rmse_blend=_clip(genome.mae_rmse_blend + rng.uniform(-0.15, 0.15), 0.0, 1.0),
            quantile_alpha=_clip(genome.quantile_alpha + rng.uniform(-0.1, 0.1), 0.01, 0.99),
            metadata=dict(genome.metadata),
        )

    def _crossover(self, parent_a: LossGenome, parent_b: LossGenome, rng: random.Random) -> LossGenome:
        return LossGenome(
            objective=parent_a.objective if rng.random() < 0.5 else parent_b.objective,
            asymmetric_weight=_pick(parent_a.asymmetric_weight, parent_b.asymmetric_weight, rng),
            huber_delta=_pick(parent_a.huber_delta, parent_b.huber_delta, rng),
            mae_rmse_blend=_pick(parent_a.mae_rmse_blend, parent_b.mae_rmse_blend, rng),
            quantile_alpha=_pick(parent_a.quantile_alpha, parent_b.quantile_alpha, rng),
            metadata={},
        )


@dataclass(frozen=True, slots=True)
class _ObjectiveSettings:
    """Validated gradient/hessian settings derived from a genome.

    @notice: |
        Validated gradient/hessian settings derived from a genome.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: settings-model
            rationale: "Clamps evolved genome values into a numerically safe range once."
    """

    alpha: float
    blend: float
    delta: float
    inverse_blend: float
    is_quantile: bool
    one_minus_alpha: float
    weight: float


def build_lightgbm_objective_params(genome: LossGenome) -> dict[str, Any]:
    """Create built-in LightGBM objective params from a LossGenome.

    @notice: |
        Create built-in LightGBM objective params from a LossGenome.

    @dev: |
        Emits only parameters LightGBM actually recognizes. ``alpha`` is the huber
        transition point for ``huber`` and the pinball level for ``quantile``.
        Asymmetric weighting and MAE/RMSE blending have no built-in equivalent, so
        they are expressed through :func:`build_custom_objective` instead of being
        passed as parameters LightGBM would silently ignore.
    """
    settings = _resolve_settings(genome)
    params: dict[str, Any] = {"objective": genome.objective}
    if genome.objective == "quantile":
        params["alpha"] = settings.alpha
    elif genome.objective == "huber":
        params["alpha"] = settings.delta
    return params


def build_custom_objective(genome: LossGenome) -> LightGBMObjective:
    """Build a LightGBM custom objective implementing the evolved loss.

    @notice: |
        Build a LightGBM custom objective implementing the evolved loss.

    @dev: |
        The returned callable works with both LightGBM entry points: ``lgb.train``
        passes ``(preds, Dataset)`` while the sklearn wrapper passes
        ``(y_true, y_pred)``. It must keep exactly two parameters because the sklearn
        wrapper switches call conventions on the callable's arity. Gradients blend
        squared error with a pseudo-Huber surrogate (``mae_rmse_blend``), switch to a
        smoothed pinball loss for the ``quantile`` objective, and scale
        over-predictions by ``asymmetric_weight``.
        Hessians are strictly positive so LightGBM leaf updates stay finite.
    """
    settings = _resolve_settings(genome)

    def custom_objective(first: Any, second: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        y_true, y_pred = _unpack_objective_args(first, second)
        return _gradient_hessian(y_true, y_pred, settings)

    custom_objective.__name__ = f"agent_k_{genome.objective}_objective"
    return custom_objective


def render_lightgbm_objective_source(genome: LossGenome) -> str:
    """Render the evolved objective as standalone source code.

    @notice: |
        Render the evolved objective as standalone source code.

    @dev: |
        Generated solutions run in an isolated interpreter that cannot import
        ``agent_k``, so the objective is emitted as a self-contained ``numpy``-only
        function. The rendered code is numerically identical to
        :func:`build_custom_objective` for the same genome.
    """
    settings = _resolve_settings(genome)
    body = (
        _QUANTILE_SOURCE_BODY.format(
            alpha=_render_float(settings.alpha),
            one_minus_alpha=_render_float(settings.one_minus_alpha),
            delta=_render_float(settings.delta),
        )
        if settings.is_quantile
        else _BLEND_SOURCE_BODY.format(
            blend=_render_float(settings.blend), inverse_blend=_render_float(settings.inverse_blend)
        )
    )
    header = _OBJECTIVE_SOURCE_HEADER.format(objective=genome.objective, delta=_render_float(settings.delta))
    footer = _OBJECTIVE_SOURCE_FOOTER.format(weight=_render_float(settings.weight), min_hessian=repr(_MIN_HESSIAN))
    return f"{header}{body}{footer}"


def _render_float(value: float) -> str:
    return repr(_round(value))


def _round(value: float) -> float:
    return round(value, _FLOAT_PRECISION)


def _resolve_settings(genome: LossGenome) -> _ObjectiveSettings:
    raw_blend = 1.0 if genome.objective in _L1_LIKE_OBJECTIVES else _clip(float(genome.mae_rmse_blend), 0.0, 1.0)
    blend = _round(raw_blend)
    alpha = _round(_clip(float(genome.quantile_alpha), _MIN_QUANTILE_ALPHA, 1.0 - _MIN_QUANTILE_ALPHA))
    return _ObjectiveSettings(
        alpha=alpha,
        blend=blend,
        delta=_round(max(float(genome.huber_delta), _MIN_DELTA)),
        inverse_blend=_round(1.0 - blend),
        is_quantile=genome.objective == "quantile",
        one_minus_alpha=_round(1.0 - alpha),
        weight=_round(max(float(genome.asymmetric_weight), _MIN_WEIGHT)),
    )


def _unpack_objective_args(first: Any, second: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if hasattr(second, "get_label"):
        return np.asarray(second.get_label(), dtype=np.float64), np.asarray(first, dtype=np.float64)
    return np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)


def _gradient_hessian(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], settings: _ObjectiveSettings
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    residual = y_pred - y_true
    scaled = residual / settings.delta
    denom = np.sqrt(1.0 + scaled * scaled)
    pseudo_grad = residual / denom
    pseudo_hess = 1.0 / (denom * denom * denom)
    if settings.is_quantile:
        side = np.where(residual < 0.0, settings.alpha, settings.one_minus_alpha)
        grad = side * pseudo_grad / settings.delta
        hess = side * pseudo_hess / settings.delta
    else:
        grad = settings.inverse_blend * residual + settings.blend * pseudo_grad
        hess = settings.inverse_blend + settings.blend * pseudo_hess
    asymmetry = np.where(residual > 0.0, settings.weight, 1.0)
    return grad * asymmetry, np.maximum(hess * asymmetry, _MIN_HESSIAN)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
