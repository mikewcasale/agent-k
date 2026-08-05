"""Loss function evolution for LightGBM objectives.

@notice: |
    Loss function evolution for LightGBM objectives, including custom Python
    objective callables that consume genome parameters LightGBM's built-in
    objectives cannot express (asymmetric penalties, blended MAE/RMSE).

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
    last-verified: 2026-08-05
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_k.evolution.framework import FitnessFn, Individual, Population

if TYPE_CHECKING:
    import numpy as np
    from lightgbm import Dataset

type LossObjective = str
"""LightGBM objective identifiers.

Built-in string identifiers: ``regression``, ``regression_l1``, ``huber``,
``quantile``. Custom callable-backed identifiers: ``asymmetric`` (weights
under/over-prediction differently) and ``blended`` (blends MAE and MSE
gradients via ``mae_rmse_blend``).
"""

ObjectiveCallable = Callable[["np.ndarray", "Dataset"], tuple["np.ndarray", "np.ndarray"]]
"""LightGBM custom objective signature returning (gradient, hessian)."""

_BUILTIN_OBJECTIVES: frozenset[str] = frozenset({"regression", "regression_l1", "huber", "quantile"})
_CUSTOM_OBJECTIVES: frozenset[str] = frozenset({"asymmetric", "blended"})
_HESSIAN_FLOOR: float = 1e-6

__all__ = (
    "LossFunctionEvolver",
    "LossGenome",
    "ObjectiveCallable",
    "build_lightgbm_objective_params",
    "make_asymmetric_objective",
    "make_blended_objective",
)


@dataclass(slots=True)
class LossGenome:
    """Genome for evolving custom objective functions.

    @notice: |
        Genome for evolving custom objective functions.

    @dev: |
        Fields ``asymmetric_weight`` and ``mae_rmse_blend`` drive custom
        Python objective callables selected via ``objective``. Fields
        ``huber_delta`` and ``quantile_alpha`` map to LightGBM's native
        ``alpha`` parameter for the ``huber`` and ``quantile`` objectives
        respectively.

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

    _objectives: tuple[str, ...] = ("regression", "regression_l1", "huber", "quantile", "asymmetric", "blended")

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
    """Create LightGBM params from a LossGenome.

    @notice: |
        Emits parameters using LightGBM's native parameter names. Built-in
        objectives yield a pure ``{"objective": str, ...}`` mapping suitable
        for ``lightgbm.train``. Custom objectives yield an
        ``{"objective": callable}`` mapping whose callable consumes genome
        fields LightGBM's built-ins cannot express.

    @dev: |
        Huber's transition point is LightGBM's ``alpha`` parameter, not
        ``huber_delta`` (which LightGBM silently ignores). Quantile
        regression also uses ``alpha``. ``asymmetric`` returns a callable
        objective driven by ``asymmetric_weight`` (>1 penalizes under-
        prediction more). ``blended`` returns a callable objective mixing
        L1 and L2 gradients per ``mae_rmse_blend`` (0.0 → pure MSE, 1.0 →
        pure MAE). Pass the returned dict straight to ``lightgbm.train``'s
        ``params``; do not merge ``asymmetric_weight`` / ``mae_rmse_blend`` /
        ``huber_delta`` yourself — LightGBM will silently discard them.
    """
    objective = genome.objective

    if objective == "huber":
        return {"objective": "huber", "alpha": genome.huber_delta}
    if objective == "quantile":
        return {"objective": "quantile", "alpha": genome.quantile_alpha}
    if objective == "asymmetric":
        return {"objective": make_asymmetric_objective(genome.asymmetric_weight)}
    if objective == "blended":
        return {"objective": make_blended_objective(genome.mae_rmse_blend)}
    if objective in _BUILTIN_OBJECTIVES:
        return {"objective": objective}

    raise ValueError(
        f"Unknown loss objective {objective!r}. Expected one of {sorted(_BUILTIN_OBJECTIVES | _CUSTOM_OBJECTIVES)}."
    )


def make_asymmetric_objective(weight: float) -> ObjectiveCallable:
    """Build a LightGBM custom objective with asymmetric squared error.

    @notice: |
        Under-predictions (pred < label) are penalized by ``weight``; over-
        predictions carry unit weight. ``weight == 1.0`` reduces to L2.

    @dev: |
        Signature matches LightGBM's callable-objective contract:
        ``(preds, train_data) -> (grad, hess)`` operating on the raw margin.
        For squared error the raw margin equals the prediction. Hessian is
        clipped away from zero so the boosting split-finder stays numerically
        stable when ``weight`` collapses toward zero.
    """
    if weight <= 0.0:
        raise ValueError(f"asymmetric weight must be positive, got {weight!r}")

    def _asymmetric(preds: np.ndarray, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        import numpy as _np

        y_true = dataset.get_label()
        residual = preds - y_true
        # weight applied when we under-predict (residual < 0)
        w = _np.where(residual < 0.0, weight, 1.0)
        grad = 2.0 * residual * w
        hess = _np.maximum(2.0 * w, _HESSIAN_FLOOR)
        return grad, hess

    return _asymmetric


def make_blended_objective(blend: float) -> ObjectiveCallable:
    """Build a LightGBM custom objective mixing L1 and L2 gradients.

    @notice: |
        ``blend=0.0`` recovers L2 (MSE), ``blend=1.0`` recovers L1 (MAE).
        Intermediate values interpolate the gradients linearly.

    @dev: |
        Signature matches LightGBM's callable-objective contract. L1's true
        second derivative is zero everywhere, so we fall back to a positive
        constant hessian for the L1 component to keep the tree learner
        numerically stable — this mirrors LightGBM's own ``regression_l1``
        implementation.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0.0, 1.0], got {blend!r}")

    def _blended(preds: np.ndarray, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        import numpy as _np

        y_true = dataset.get_label()
        residual = preds - y_true
        l1_grad = _np.sign(residual)
        l2_grad = 2.0 * residual
        grad = blend * l1_grad + (1.0 - blend) * l2_grad
        hess = _np.full_like(residual, blend + 2.0 * (1.0 - blend), dtype=_np.float64)
        _np.maximum(hess, _HESSIAN_FLOOR, out=hess)
        return grad, hess

    return _blended


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
