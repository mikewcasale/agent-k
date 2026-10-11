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

import ast
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from agent_k.evolution.framework import FitnessFn, Individual, Population

type LossObjective = str
"""LightGBM objective identifiers."""

type GradHess = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""Gradient and hessian arrays returned by a LightGBM custom objective."""

type LightGBMObjective = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], GradHess]
"""Callable LightGBM objective with the scikit-learn ``(y_true, y_pred)`` signature."""

__all__ = (
    "ALPHA_BOUNDS",
    "ALPHA_OBJECTIVES",
    "CUSTOM_ASYMMETRIC_OBJECTIVE",
    "DEFAULT_ALPHA",
    "MIN_HESSIAN",
    "LossFunctionEvolver",
    "LossGenome",
    "apply_lightgbm_objective",
    "build_lightgbm_objective",
    "build_lightgbm_objective_params",
    "normalize_lightgbm_loss_params",
    "objective_alpha_bounds",
    "sample_objective_alpha",
)

MIN_HESSIAN: Final[float] = 1e-6
"""Hessian floor. A non-positive hessian collapses LightGBM to a constant prediction."""

CUSTOM_ASYMMETRIC_OBJECTIVE: Final[str] = "custom_asymmetric"
"""Genome objective that compiles to a callable instead of a built-in LightGBM loss."""

ALPHA_OBJECTIVES: Final[frozenset[str]] = frozenset({"huber", "quantile"})
"""Built-in LightGBM objectives parameterized by ``alpha``."""

DEFAULT_ALPHA: Final[dict[str, float]] = {"huber": 1.0, "quantile": 0.5}
"""Neutral ``alpha`` per objective. LightGBM otherwise defaults to 0.9 for both."""

ALPHA_BOUNDS: Final[dict[str, tuple[float, float]]] = {"huber": (0.1, 10.0), "quantile": (0.05, 0.95)}
"""Valid ``alpha`` range per objective. LightGBM hard-fails outside ``(0, 1)`` for quantile."""

_FICTIONAL_ALPHA_KEYS: Final[dict[str, str]] = {"huber": "huber_delta", "quantile": "quantile_alpha"}
_LIGHTGBM_ESTIMATORS: Final[frozenset[str]] = frozenset({"LGBMClassifier", "LGBMModel", "LGBMRanker", "LGBMRegressor"})


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

    _objectives: tuple[str, ...] = ("regression", "regression_l1", "huber", "quantile", CUSTOM_ASYMMETRIC_OBJECTIVE)

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
            huber_delta=self._rng.uniform(*ALPHA_BOUNDS["huber"]),
            mae_rmse_blend=self._rng.uniform(0.0, 1.0),
            quantile_alpha=self._rng.uniform(*ALPHA_BOUNDS["quantile"]),
        )

    def _mutate(self, genome: LossGenome, rng: random.Random) -> LossGenome:
        return LossGenome(
            objective=rng.choice(self._objectives) if rng.random() < 0.2 else genome.objective,
            asymmetric_weight=_clip(genome.asymmetric_weight + rng.uniform(-0.2, 0.2), 0.1, 3.0),
            huber_delta=_clip(genome.huber_delta + rng.uniform(-0.3, 0.3), *ALPHA_BOUNDS["huber"]),
            mae_rmse_blend=_clip(genome.mae_rmse_blend + rng.uniform(-0.15, 0.15), 0.0, 1.0),
            quantile_alpha=_clip(genome.quantile_alpha + rng.uniform(-0.1, 0.1), *ALPHA_BOUNDS["quantile"]),
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
class _ParamEntry:
    """One parameter at a LightGBM parameter site, with the node naming it.

    @notice: |
        One parameter at a LightGBM parameter site, with the node naming it.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: ast-node-pair
            rationale: "Keeps the key node beside its value for source-accurate rewrites."
            violations: "Re-walking the tree per parameter loses the key positions."
    """

    key: ast.keyword | ast.Constant
    value: ast.expr


@dataclass(frozen=True, slots=True)
class _ParamScope:
    """One LightGBM parameter site - estimator keywords or a params dict.

    @notice: |
        One LightGBM parameter site - estimator keywords or a params dict.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: source-span-model
            rationale: "Carries byte spans so rewrites preserve surrounding formatting."
            violations: "Regex rewrites of loss parameters hit unrelated estimators."
    """

    is_dict: bool
    objective: str | None
    objective_span: tuple[int, int] | None
    alpha_span: tuple[int, int] | None
    fictional_key_spans: dict[str, tuple[int, int]]
    insert_at: int


def build_lightgbm_objective_params(genome: LossGenome) -> dict[str, Any]:
    """Create LightGBM params from a LossGenome.

    @notice: |
        Create LightGBM params from a LossGenome.

    @dev: |
        Only emits parameters LightGBM actually reads. ``huber`` and ``quantile``
        are parameterized by ``alpha`` (LightGBM has no ``huber_delta`` or
        ``quantile_alpha`` parameter and silently ignores both). The asymmetric
        blend genes have no built-in equivalent, so they compile to a callable
        objective instead.
    """
    if genome.objective == CUSTOM_ASYMMETRIC_OBJECTIVE:
        return {"objective": build_lightgbm_objective(genome)}

    params: dict[str, Any] = {"objective": genome.objective}
    if genome.objective == "quantile":
        params["alpha"] = _clip(genome.quantile_alpha, *ALPHA_BOUNDS["quantile"])
    elif genome.objective == "huber":
        params["alpha"] = _clip(genome.huber_delta, *ALPHA_BOUNDS["huber"])
    return params


def build_lightgbm_objective(genome: LossGenome) -> LightGBMObjective:
    """Compile a genome into a callable LightGBM objective.

    @notice: |
        Compile a genome into a callable LightGBM objective.

    @dev: |
        Blends squared error with Huber's absolute error
        (``mae_rmse_blend`` 0 is pure L2, 1 is pure Huber) and scales the
        over-prediction side by ``asymmetric_weight``.

        The Huber branch uses the IRLS weight ``delta / |residual|`` rather than
        Huber's exact hessian, which is 0 outside the delta band. A hessian of 0
        - or one merely floored to MIN_HESSIAN - leaves every split below
        LightGBM's ``min_sum_hessian_in_leaf`` (1e-3), so the booster grows no
        trees and predicts a constant.
    """
    delta = _clip(genome.huber_delta, *ALPHA_BOUNDS["huber"])
    blend = _clip(genome.mae_rmse_blend, 0.0, 1.0)
    weight = max(genome.asymmetric_weight, MIN_HESSIAN)

    def objective(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> GradHess:
        residual = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
        magnitude = np.abs(residual)
        grad_huber = np.clip(residual, -delta, delta)
        hess_huber = delta / np.maximum(magnitude, delta)
        grad = (1.0 - blend) * residual + blend * grad_huber
        hess = (1.0 - blend) * np.ones_like(residual) + blend * hess_huber
        scale = np.where(residual > 0.0, weight, 1.0)
        return grad * scale, np.maximum(hess * scale, MIN_HESSIAN)

    return objective


def objective_alpha_bounds(code: str) -> tuple[float, float] | None:
    """Return the valid ``alpha`` range for the LightGBM objective used in ``code``.

    @notice: |
        Return the valid alpha range for the LightGBM objective used in code.

    @dev: |
        Returns None when the code declares no alpha-parameterized objective, so
        callers can fall back to their generic bounds. Mutating ``alpha`` past
        LightGBM's limits aborts training (quantile requires ``0 < alpha < 1``).
    """
    for scope in _collect_param_scopes(code):
        if scope.objective in ALPHA_BOUNDS:
            return ALPHA_BOUNDS[scope.objective]
    return None


def sample_objective_alpha(objective: str, rng: random.Random) -> float | None:
    """Draw an ``alpha`` inside the objective's valid range, or None when unused.

    @notice: |
        Draw an alpha inside the objective's valid range, or None when unused.

    @dev: |
        See module for behavior details and invariants.
    """
    bounds = ALPHA_BOUNDS.get(objective)
    if bounds is None:
        return None
    return round(rng.uniform(*bounds), 4)


def normalize_lightgbm_loss_params(code: str) -> tuple[str, list[str]]:
    """Make LightGBM loss parameters in ``code`` effective instead of ignored.

    @notice: |
        Make LightGBM loss parameters in code effective instead of ignored.

    @dev: |
        Renames ``huber_delta``/``quantile_alpha`` (which LightGBM ignores) to
        ``alpha``, and gives alpha-parameterized objectives an explicit alpha so
        they do not inherit LightGBM's 0.9 default - which turns ``quantile``
        into a 90th-percentile predictor. Returns the code unchanged when it
        parses to nothing actionable or the rewrite would not re-parse.
    """
    edits: list[tuple[int, int, str]] = []
    notes: list[str] = []
    for scope in _collect_param_scopes(code):
        objective = scope.objective
        if objective is None or objective not in ALPHA_OBJECTIVES:
            continue
        if scope.alpha_span is not None:
            continue
        fictional = scope.fictional_key_spans.get(_FICTIONAL_ALPHA_KEYS[objective])
        if fictional is not None:
            edits.append((*fictional, '"alpha"' if scope.is_dict else "alpha"))
            notes.append(f"renamed {_FICTIONAL_ALPHA_KEYS[objective]} to alpha for objective={objective}")
            continue
        alpha = DEFAULT_ALPHA[objective]
        suffix = f', "alpha": {alpha}' if scope.is_dict else f", alpha={alpha}"
        edits.append((scope.insert_at, scope.insert_at, suffix))
        notes.append(f"set alpha={alpha} for objective={objective}")

    if not edits:
        return code, notes
    updated = _apply_edits(code, edits)
    if not _parses(updated):
        return code, []
    return updated, notes


def apply_lightgbm_objective(code: str, objective: str, alpha: float | None) -> tuple[str, list[str]]:
    """Set the LightGBM objective in ``code`` and keep its ``alpha`` consistent.

    @notice: |
        Set the LightGBM objective in code and keep its alpha consistent.

    @dev: |
        Objective and alpha have to move together: leaving a huber delta behind
        when switching to ``quantile`` aborts training, and switching to
        ``quantile`` without an alpha silently predicts the 90th percentile.
    """
    edits: list[tuple[int, int, str]] = []
    notes: list[str] = []
    for scope in _collect_param_scopes(code):
        if scope.objective is None or scope.objective_span is None:
            continue
        edits.append((*scope.objective_span, f'"{objective}"'))
        if alpha is None:
            continue
        if scope.alpha_span is not None:
            edits.append((*scope.alpha_span, f"{alpha}"))
        else:
            suffix = f', "alpha": {alpha}' if scope.is_dict else f", alpha={alpha}"
            edits.append((scope.insert_at, scope.insert_at, suffix))
        notes.append(f"objective={objective} alpha={alpha}")

    if not edits:
        return code, notes
    updated = _apply_edits(code, edits)
    if not _parses(updated):
        return code, []
    return updated, notes


def _collect_param_scopes(code: str) -> list[_ParamScope]:
    if "lightgbm" not in code and "lgb" not in code:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    offsets = _line_offsets(code)
    scopes: list[_ParamScope] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) in _LIGHTGBM_ESTIMATORS:
            keywords: dict[str, _ParamEntry] = {
                keyword.arg: _ParamEntry(key=keyword, value=keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            scope = _build_scope(keywords, offsets, is_dict=False)
        elif isinstance(node, ast.Dict):
            keywords = {
                key.value: _ParamEntry(key=key, value=value)
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            scope = _build_scope(keywords, offsets, is_dict=True)
        else:
            continue
        if scope is not None:
            scopes.append(scope)
    return scopes


def _build_scope(entries: dict[str, _ParamEntry], offsets: list[int], *, is_dict: bool) -> _ParamScope | None:
    values = {name: entry.value for name, entry in entries.items()}
    objective_node = values.get("objective")
    if objective_node is None:
        return None
    if not isinstance(objective_node, ast.Constant) or not isinstance(objective_node.value, str):
        # A callable objective; its parameters live inside the callable.
        return None

    alpha_node = values.get("alpha")
    return _ParamScope(
        is_dict=is_dict,
        objective=objective_node.value,
        objective_span=_span(objective_node, offsets),
        alpha_span=_span(alpha_node, offsets) if isinstance(alpha_node, ast.expr) else None,
        fictional_key_spans={
            name: _key_span(entries[name].key, offsets) for name in _FICTIONAL_ALPHA_KEYS.values() if name in entries
        },
        insert_at=_span(objective_node, offsets)[1],
    )


def _line_offsets(code: str) -> list[int]:
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _span(node: ast.expr | ast.keyword, offsets: list[int]) -> tuple[int, int]:
    end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
    end_col = node.end_col_offset if node.end_col_offset is not None else node.col_offset
    return offsets[node.lineno - 1] + node.col_offset, offsets[end_lineno - 1] + end_col


def _key_span(key: ast.keyword | ast.Constant, offsets: list[int]) -> tuple[int, int]:
    if isinstance(key, ast.Constant):
        return _span(key, offsets)
    start = offsets[key.lineno - 1] + key.col_offset
    return start, start + len(key.arg or "")


def _apply_edits(code: str, edits: list[tuple[int, int, str]]) -> str:
    updated = code
    for start, end, replacement in sorted(edits, key=lambda edit: edit[0], reverse=True):
        updated = f"{updated[:start]}{replacement}{updated[end:]}"
    return updated


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _pick(left: float, right: float, rng: random.Random) -> float:
    return left if rng.random() < 0.5 else right
