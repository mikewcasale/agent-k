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
    last-verified: 2026-07-09
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAliasType

import numpy as np

from agent_k.evolution.framework import FitnessFn, Individual, Population

if TYPE_CHECKING:
    from numpy.typing import NDArray

type LossObjective = str
"""LightGBM objective identifiers (built-ins) plus the ``"asymmetric"`` sentinel used by the custom fobj."""

ObjectiveCallable = TypeAliasType(
    "ObjectiveCallable",
    Callable[["NDArray[np.floating[Any]]", Any], tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]],
)
"""Signature accepted by ``lightgbm.train(..., fobj=...)``: ``(preds, train_data) -> (grad, hess)``."""

__all__ = (
    "ASYMMETRIC_OBJECTIVE",
    "BUILTIN_OBJECTIVES",
    "LossFunctionEvolver",
    "LossGenome",
    "ObjectiveCallable",
    "build_custom_lightgbm_objective",
    "build_lightgbm_objective_params",
)

ASYMMETRIC_OBJECTIVE: str = "asymmetric"
"""Sentinel objective name that maps to the custom fobj instead of a LightGBM built-in."""

BUILTIN_OBJECTIVES: frozenset[str] = frozenset({"regression", "regression_l1", "huber", "quantile"})
"""LightGBM built-in objectives recognised by :func:`build_lightgbm_objective_params`."""

_MAE_HESS_EPS: float = 1.0
"""Constant hessian used for the L1 component of the asymmetric fobj.

@dev: |
    A pure L1 loss has zero second derivative almost everywhere, which stalls LightGBM's
    Newton step. LightGBM's own ``regression_l1`` objective uses a constant hessian of 1;
    we mirror that so a fully MAE-weighted blend (blend=0) still produces well-scaled
    gradients.
"""


@dataclass(slots=True)
class LossGenome:
    """Genome for evolving custom objective functions.

    @notice: |
        Genome for evolving custom objective functions.

    @dev: |
        ``objective`` is either one of :data:`BUILTIN_OBJECTIVES` (mapped straight through
        to LightGBM's ``objective`` parameter) or :data:`ASYMMETRIC_OBJECTIVE`, which
        activates the custom fobj built from ``asymmetric_weight`` and ``mae_rmse_blend``.

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

    _objectives: tuple[str, ...] = ("regression", "regression_l1", "huber", "quantile", ASYMMETRIC_OBJECTIVE)

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


def build_lightgbm_objective_params(genome: LossGenome) -> dict[str, Any]:
    """Return a LightGBM params dict for the given genome.

    @notice: |
        Return a LightGBM params dict for the given genome.

    @dev: |
        The returned dict is ready to pass straight to ``lightgbm.train`` — for
        :data:`ASYMMETRIC_OBJECTIVE` the ``objective`` key holds a callable produced by
        :func:`build_custom_lightgbm_objective` (LightGBM >= 4 replaced the legacy
        ``fobj=`` kwarg with an objective-callable in ``params``); for built-ins it
        holds the objective's string name plus the ``alpha`` parameter where relevant.

        Fields that only apply to the custom fobj (``asymmetric_weight``,
        ``mae_rmse_blend``) are intentionally omitted from the built-in branch so
        LightGBM does not warn "Unknown parameter". The ``huber`` and ``quantile``
        objectives both use LightGBM's ``alpha`` parameter — the historical
        ``huber_delta`` key was silently ignored by LightGBM
        ("``[LightGBM] [Warning] Unknown parameter: huber_delta``"), so every evolved
        Huber delta had zero effect on training until this fix.
    """
    if genome.objective == ASYMMETRIC_OBJECTIVE:
        fobj = build_custom_lightgbm_objective(genome)
        assert fobj is not None
        return {"objective": fobj}
    if genome.objective not in BUILTIN_OBJECTIVES:
        return {"objective": genome.objective}
    params: dict[str, Any] = {"objective": genome.objective}
    if genome.objective == "huber":
        params["alpha"] = _clip(genome.huber_delta, 1e-8, 1e8)
    elif genome.objective == "quantile":
        params["alpha"] = _clip(genome.quantile_alpha, 1e-8, 1.0 - 1e-8)
    return params


def build_custom_lightgbm_objective(genome: LossGenome) -> ObjectiveCallable | None:
    """Return a LightGBM ``fobj`` callable for a custom-loss genome, else ``None``.

    @notice: |
        Return a LightGBM ``fobj`` callable for a custom-loss genome, else ``None``.

    @dev: |
        The callable implements an asymmetric MAE/RMSE blend suitable for
        ``lightgbm.train(params, ..., fobj=...)``.

        - ``mae_rmse_blend`` linearly blends between MAE (0.0) and MSE (1.0) gradients.
        - ``asymmetric_weight`` multiplies the gradient/hessian for over-predictions
          (``pred - y > 0``); values above 1 push predictions downward, values below 1
          upward.

        Returns ``None`` when the genome selects a LightGBM built-in objective so the
        caller can rely on ``fobj is None`` to mean "use the built-in".
    """
    if genome.objective != ASYMMETRIC_OBJECTIVE:
        return None
    weight = _clip(genome.asymmetric_weight, 1e-3, 1e3)
    blend = _clip(genome.mae_rmse_blend, 0.0, 1.0)
    return _make_asymmetric_fobj(weight=weight, blend=blend)


def _make_asymmetric_fobj(*, weight: float, blend: float) -> ObjectiveCallable:
    def fobj(
        preds: NDArray[np.floating[Any]], train_data: Any
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        labels = np.asarray(train_data.get_label(), dtype=np.float64)
        preds_arr = np.asarray(preds, dtype=np.float64).reshape(labels.shape)
        residual = preds_arr - labels
        mse_grad = residual
        mse_hess = np.ones_like(residual)
        mae_grad = np.sign(residual)
        mae_hess = np.full_like(residual, _MAE_HESS_EPS)
        grad = blend * mse_grad + (1.0 - blend) * mae_grad
        hess = blend * mse_hess + (1.0 - blend) * mae_hess
        over = residual > 0.0
        grad = np.where(over, grad * weight, grad)
        hess = np.where(over, hess * weight, hess)
        return grad, hess

    return fobj


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
