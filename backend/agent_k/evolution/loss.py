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
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from agent_k.evolution.framework import FitnessFn, Individual, Population

if TYPE_CHECKING:
    from collections.abc import Callable

type LossObjective = str
"""LightGBM objective identifiers."""

type ObjectiveFamily = Literal["regression", "binary", "multiclass", "ranking"]
"""Problem families that partition LightGBM objectives by target domain."""

type CustomObjective = Callable[[Any, Any], tuple[Any, Any]]
"""LightGBM custom objective returning ``(grad, hess)`` for ``(y_pred, dataset)``."""

__all__ = (
    "LIGHTGBM_OBJECTIVE_FAMILIES",
    "CustomObjective",
    "LossFunctionEvolver",
    "LossGenome",
    "ObjectiveFamily",
    "alternative_objectives",
    "build_custom_objective",
    "build_lightgbm_objective_params",
    "canonical_objective",
    "family_objectives",
    "objective_family",
)

LIGHTGBM_OBJECTIVE_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "regression": ("regression", "regression_l1", "huber", "fair", "quantile"),
    "binary": ("binary", "cross_entropy"),
    "multiclass": ("multiclass", "multiclassova"),
    "ranking": ("lambdarank", "rank_xendcg"),
}
"""Interchangeable LightGBM objectives per problem family.

Objectives inside a family accept the same target domain, so swapping one for another keeps a
candidate program runnable. Swapping across families does not: a regression objective on a
multiclass booster raises ``LightGBMError``, and a ``quantile`` objective on a binary classifier
collapses predictions to a constant. Objectives that constrain the target domain (``poisson``,
``gamma``, ``tweedie``, ``mape``) are deliberately absent: they require non-negative, strictly
positive, or non-zero targets, so they are never produced as a substitute.
"""

_OBJECTIVE_ALIASES: Final[dict[str, str]] = {
    "regression_l2": "regression",
    "l2": "regression",
    "mean_squared_error": "regression",
    "mse": "regression",
    "l2_root": "regression",
    "root_mean_squared_error": "regression",
    "rmse": "regression",
    "l1": "regression_l1",
    "mean_absolute_error": "regression_l1",
    "mae": "regression_l1",
    "xentropy": "cross_entropy",
    "softmax": "multiclass",
    "multiclass_ova": "multiclassova",
    "ova": "multiclassova",
    "ovr": "multiclassova",
    "xendcg": "rank_xendcg",
    "xe_ndcg": "rank_xendcg",
    "xe_ndcg_mart": "rank_xendcg",
    "xendcg_mart": "rank_xendcg",
}

_CANONICAL_FAMILIES: Final[dict[str, str]] = {
    objective: family for family, objectives in LIGHTGBM_OBJECTIVE_FAMILIES.items() for objective in objectives
}

_MIN_HESSIAN: Final[float] = 1e-6
"""Hessian floor keeping LightGBM leaf updates finite."""


def canonical_objective(objective: str) -> str | None:
    """Resolve a LightGBM objective name or alias to its canonical spelling.

    @notice: |
        Resolve a LightGBM objective name or alias to its canonical spelling.

    @dev: |
        Quoting and case from source code are stripped first. Returns ``None`` for objectives
        outside the known families, including callables and narrow-domain objectives, so callers
        can decline to reason about them.
    """
    normalized = objective.strip().strip("\"'").lower()
    resolved = _OBJECTIVE_ALIASES.get(normalized, normalized)
    return resolved if resolved in _CANONICAL_FAMILIES else None


def objective_family(objective: str) -> str | None:
    """Resolve the problem family of a LightGBM objective name or alias.

    @notice: |
        Resolve the problem family of a LightGBM objective name or alias.

    @dev: |
        Returns ``None`` for unknown objectives so search operators can leave them untouched
        instead of guessing a target domain.
    """
    canonical = canonical_objective(objective)
    return None if canonical is None else _CANONICAL_FAMILIES[canonical]


def family_objectives(family: str) -> tuple[str, ...]:
    """Return the interchangeable objectives for a problem family.

    @notice: |
        Return the interchangeable objectives for a problem family.

    @dev: |
        Unknown families yield an empty tuple rather than raising, so search operators degrade
        to leaving the objective untouched.
    """
    return LIGHTGBM_OBJECTIVE_FAMILIES.get(family, ())


def alternative_objectives(objective: str) -> tuple[str, ...]:
    """Return same-family objectives that can safely replace ``objective``.

    @notice: |
        Return same-family objectives that can safely replace ``objective``.

    @dev: |
        The canonical form of the current objective is excluded so callers always get a genuine
        alternative. An unknown objective yields an empty tuple.
    """
    canonical = canonical_objective(objective)
    if canonical is None:
        return ()
    return tuple(name for name in family_objectives(_CANONICAL_FAMILIES[canonical]) if name != canonical)


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
    family: ObjectiveFamily = "regression"
    asymmetric_weight: float = 1.0
    huber_delta: float = 1.0
    mae_rmse_blend: float = 0.5
    quantile_alpha: float = 0.5
    fair_c: float = 1.0
    num_class: int = 1
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

    def __init__(
        self,
        fitness_fn: FitnessFn[LossGenome],
        *,
        family: ObjectiveFamily = "regression",
        num_class: int = 1,
        population_size: int = 16,
        rng: random.Random | None = None,
    ) -> None:
        objectives = family_objectives(family)
        if not objectives:
            message = f"Unknown LightGBM objective family: {family!r}"
            raise ValueError(message)
        self._fitness_fn = fitness_fn
        self._family: ObjectiveFamily = family
        self._num_class = max(1, num_class)
        self._objectives = objectives
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
            family=self._family,
            asymmetric_weight=self._rng.uniform(0.5, 2.0),
            huber_delta=self._rng.uniform(0.5, 5.0),
            mae_rmse_blend=self._rng.uniform(0.0, 1.0),
            quantile_alpha=self._rng.uniform(0.05, 0.95),
            fair_c=self._rng.uniform(0.5, 5.0),
            num_class=self._num_class,
        )

    def _mutate(self, genome: LossGenome, rng: random.Random) -> LossGenome:
        return LossGenome(
            objective=rng.choice(self._objectives) if rng.random() < 0.2 else genome.objective,
            family=self._family,
            asymmetric_weight=_clip(genome.asymmetric_weight + rng.uniform(-0.2, 0.2), 0.1, 3.0),
            huber_delta=_clip(genome.huber_delta + rng.uniform(-0.3, 0.3), 0.1, 10.0),
            mae_rmse_blend=_clip(genome.mae_rmse_blend + rng.uniform(-0.15, 0.15), 0.0, 1.0),
            quantile_alpha=_clip(genome.quantile_alpha + rng.uniform(-0.1, 0.1), 0.01, 0.99),
            fair_c=_clip(genome.fair_c + rng.uniform(-0.3, 0.3), 0.1, 10.0),
            num_class=self._num_class,
            metadata=dict(genome.metadata),
        )

    def _crossover(self, parent_a: LossGenome, parent_b: LossGenome, rng: random.Random) -> LossGenome:
        return LossGenome(
            objective=parent_a.objective if rng.random() < 0.5 else parent_b.objective,
            family=self._family,
            asymmetric_weight=_pick(parent_a.asymmetric_weight, parent_b.asymmetric_weight, rng),
            huber_delta=_pick(parent_a.huber_delta, parent_b.huber_delta, rng),
            mae_rmse_blend=_pick(parent_a.mae_rmse_blend, parent_b.mae_rmse_blend, rng),
            quantile_alpha=_pick(parent_a.quantile_alpha, parent_b.quantile_alpha, rng),
            fair_c=_pick(parent_a.fair_c, parent_b.fair_c, rng),
            num_class=self._num_class,
            metadata={},
        )


def build_lightgbm_objective_params(genome: LossGenome) -> dict[str, Any]:
    """Create LightGBM booster params from a LossGenome.

    @notice: |
        Create LightGBM booster params from a LossGenome.

    @dev: |
        Only parameters LightGBM actually reads are emitted. ``huber`` and ``quantile`` are both
        controlled by ``alpha`` (LightGBM has no ``huber_delta`` parameter, so emitting one is a
        silent no-op), ``fair`` is controlled by ``fair_c``, and multiclass objectives require
        ``num_class``. The genome's ``asymmetric_weight`` and ``mae_rmse_blend`` genes shape the
        callable from :func:`build_custom_objective` instead, since LightGBM has no native
        parameter for either.
    """
    objective = canonical_objective(genome.objective) or genome.objective
    params: dict[str, Any] = {"objective": objective}
    if objective == "quantile":
        params["alpha"] = genome.quantile_alpha
    elif objective == "huber":
        params["alpha"] = genome.huber_delta
    elif objective == "fair":
        params["fair_c"] = genome.fair_c
    if objective in LIGHTGBM_OBJECTIVE_FAMILIES["multiclass"]:
        params["num_class"] = max(2, genome.num_class)
    return params


def build_custom_objective(genome: LossGenome) -> CustomObjective:
    """Build a LightGBM custom objective callable from a LossGenome.

    @notice: |
        Build a LightGBM custom objective callable from a LossGenome.

    @dev: |
        The returned callable matches LightGBM's ``lgb.train`` contract: it takes the raw scores
        and the training :class:`lightgbm.Dataset` and returns ``(grad, hess)``. Pass it as
        ``params["objective"]`` (LightGBM 4.x removed the ``fobj`` argument).

        Regression blends a squared-error term with a pseudo-Huber term via ``mae_rmse_blend``
        (0 is pure L2, 1 is pure pseudo-Huber), so the hessian stays strictly positive where
        LightGBM's native ``huber`` zeroes it outside the delta band. Binary uses weighted
        logistic loss and multiclass uses weighted softmax cross-entropy.

        ``asymmetric_weight`` reweights the costly side of each loss: under-predictions for
        regression, the positive class for binary, and — above 1.0 — the rarer classes for
        multiclass, interpolating from uniform to fully inverse-frequency weighting.

    @raises ValueError: If the genome family has no custom-objective formulation.
    """
    if genome.family == "regression":
        return _regression_objective(genome)
    if genome.family == "binary":
        return _binary_objective(genome)
    if genome.family == "multiclass":
        return _multiclass_objective(genome)
    message = (
        f"No custom objective formulation for family {genome.family!r}; "
        f"use build_lightgbm_objective_params for the native objective instead"
    )
    raise ValueError(message)


def _regression_objective(genome: LossGenome) -> CustomObjective:
    blend = _clip(genome.mae_rmse_blend, 0.0, 1.0)
    delta = max(genome.huber_delta, 1e-3)
    weight = max(genome.asymmetric_weight, 1e-3)

    def objective(y_pred: Any, dataset: Any) -> tuple[Any, Any]:
        y_true = np.asarray(dataset.get_label(), dtype=np.float64)
        residual = np.asarray(y_pred, dtype=np.float64).ravel() - y_true
        scaled = residual / delta
        denominator = np.sqrt(1.0 + scaled * scaled)
        grad = (1.0 - blend) * residual + blend * (residual / denominator)
        hess = (1.0 - blend) + blend / denominator**3
        weights = np.where(residual < 0.0, weight, 1.0)
        return grad * weights, np.maximum(hess * weights, _MIN_HESSIAN)

    return objective


def _binary_objective(genome: LossGenome) -> CustomObjective:
    weight = max(genome.asymmetric_weight, 1e-3)

    def objective(y_pred: Any, dataset: Any) -> tuple[Any, Any]:
        y_true = np.asarray(dataset.get_label(), dtype=np.float64)
        probability = _sigmoid(np.asarray(y_pred, dtype=np.float64).ravel())
        weights = np.where(y_true > 0.5, weight, 1.0)
        grad = weights * (probability - y_true)
        hess = weights * probability * (1.0 - probability)
        return grad, np.maximum(hess, _MIN_HESSIAN)

    return objective


def _multiclass_objective(genome: LossGenome) -> CustomObjective:
    num_class = max(2, genome.num_class)
    balance = _clip(genome.asymmetric_weight - 1.0, 0.0, 1.0)

    def objective(y_pred: Any, dataset: Any) -> tuple[Any, Any]:
        labels = np.asarray(dataset.get_label(), dtype=np.int64)
        scores = np.asarray(y_pred, dtype=np.float64).reshape(len(labels), num_class)
        probability = _softmax(scores)
        one_hot = np.zeros_like(probability)
        one_hot[np.arange(len(labels)), labels] = 1.0
        weights = _class_balance_weights(labels, num_class=num_class, balance=balance)[:, None]
        grad = weights * (probability - one_hot)
        hess = weights * 2.0 * probability * (1.0 - probability)
        return grad, np.maximum(hess, _MIN_HESSIAN)

    return objective


def _class_balance_weights(labels: Any, *, num_class: int, balance: float) -> Any:
    """Per-sample weights interpolating between uniform and inverse-frequency balancing."""
    counts = np.bincount(labels, minlength=num_class).astype(np.float64)
    frequency = np.maximum(counts, 1.0) / float(len(labels))
    class_weights = ((1.0 / num_class) / frequency) ** balance
    weights = class_weights[labels]
    return weights / weights.mean()


def _sigmoid(scores: Any) -> Any:
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(scores, -50.0, 50.0)))


def _softmax(scores: Any) -> Any:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
