"""Generic evaluation-metric resolution for AGENT-K.

@notice: |
    Generic evaluation-metric resolution for AGENT-K.

@dev: |
    Maps free-text platform metric names (camelCase slugs, spelled-out
    phrases, parenthesised acronyms) onto EvaluationMetric plus the
    optimisation direction, and owns the canonical classification and
    probability metric sets shared by policy and codegen logic.

@graph:
    id: agent_k.core.metrics
    provides:
        - agent_k.core.metrics
    pattern: resolver

@agent-guidance:
    do:
        - "Use agent_k.core.metrics as the canonical home for this capability."
        - "Add new metric spellings to the phrase/token tables instead of inline branches."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Add competition-specific metric names; keep entries generic per problem type."

@human-review:
    last-verified: 2026-10-01
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import logfire

from .models import EvaluationMetric

if TYPE_CHECKING:
    from .types import MetricDirection

__all__ = (
    "CLASSIFICATION_METRICS",
    "DEFAULT_METRIC",
    "METRIC_DIRECTIONS",
    "PROBABILITY_METRICS",
    "MetricResolution",
    "normalize_metric_text",
    "resolve_evaluation_metric",
)

DEFAULT_METRIC: Final[EvaluationMetric] = EvaluationMetric.ACCURACY

METRIC_DIRECTIONS: Final[dict[EvaluationMetric, MetricDirection]] = {
    EvaluationMetric.ACCURACY: "maximize",
    EvaluationMetric.AUC: "maximize",
    EvaluationMetric.F1: "maximize",
    EvaluationMetric.MAP: "maximize",
    EvaluationMetric.NDCG: "maximize",
    EvaluationMetric.LOG_LOSS: "minimize",
    EvaluationMetric.RMSE: "minimize",
    EvaluationMetric.MAE: "minimize",
    EvaluationMetric.RMSLE: "minimize",
}
"""Optimisation direction for every supported evaluation metric."""

CLASSIFICATION_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {EvaluationMetric.ACCURACY, EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS, EvaluationMetric.F1}
)
"""Metrics whose problem type is classification rather than regression or ranking."""

PROBABILITY_METRICS: Final[frozenset[EvaluationMetric]] = frozenset({EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS})
"""Metrics scored against predicted probabilities instead of hard labels."""

_CAMEL_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

_METRIC_PHRASES: Final[tuple[tuple[str, EvaluationMetric], ...]] = (
    # Logarithmic error variants must precede the plain squared-error phrases.
    ("root mean squared logarithmic error", EvaluationMetric.RMSLE),
    ("mean squared logarithmic error", EvaluationMetric.RMSLE),
    ("root mean squared log error", EvaluationMetric.RMSLE),
    ("logarithmic error", EvaluationMetric.RMSLE),
    ("root mean squared scaled error", EvaluationMetric.RMSE),
    ("root mean squared percentage error", EvaluationMetric.RMSE),
    ("root mean squared error", EvaluationMetric.RMSE),
    ("root mean square error", EvaluationMetric.RMSE),
    ("mean squared error", EvaluationMetric.RMSE),
    ("mean square error", EvaluationMetric.RMSE),
    ("squared error", EvaluationMetric.RMSE),
    ("mean absolute percentage error", EvaluationMetric.MAE),
    ("mean absolute error", EvaluationMetric.MAE),
    ("median absolute error", EvaluationMetric.MAE),
    ("absolute error", EvaluationMetric.MAE),
    ("continuous ranked probability score", EvaluationMetric.MAE),
    ("pinball loss", EvaluationMetric.MAE),
    ("quantile loss", EvaluationMetric.MAE),
    ("negative log likelihood", EvaluationMetric.LOG_LOSS),
    ("cross entropy", EvaluationMetric.LOG_LOSS),
    ("log loss", EvaluationMetric.LOG_LOSS),
    ("area under the precision recall curve", EvaluationMetric.MAP),
    ("precision recall curve", EvaluationMetric.MAP),
    ("mean average precision", EvaluationMetric.MAP),
    ("average precision", EvaluationMetric.MAP),
    ("receiver operating characteristic", EvaluationMetric.AUC),
    ("area under the roc curve", EvaluationMetric.AUC),
    ("area under roc curve", EvaluationMetric.AUC),
    ("area under the curve", EvaluationMetric.AUC),
    ("area under curve", EvaluationMetric.AUC),
    ("normalized discounted cumulative gain", EvaluationMetric.NDCG),
    ("discounted cumulative gain", EvaluationMetric.NDCG),
    ("intersection over union", EvaluationMetric.F1),
    ("matthews correlation", EvaluationMetric.F1),
    ("dice coefficient", EvaluationMetric.F1),
    ("dice similarity", EvaluationMetric.F1),
    ("f1 score", EvaluationMetric.F1),
    ("f score", EvaluationMetric.F1),
    ("categorization accuracy", EvaluationMetric.ACCURACY),
    ("classification accuracy", EvaluationMetric.ACCURACY),
    ("balanced accuracy", EvaluationMetric.ACCURACY),
    ("accuracy", EvaluationMetric.ACCURACY),
)
"""Ordered metric phrases, most specific first, matched against normalised text."""

_METRIC_TOKENS: Final[dict[str, EvaluationMetric]] = {
    "rmsle": EvaluationMetric.RMSLE,
    "msle": EvaluationMetric.RMSLE,
    "rmse": EvaluationMetric.RMSE,
    "mse": EvaluationMetric.RMSE,
    "rmspe": EvaluationMetric.RMSE,
    "rmsse": EvaluationMetric.RMSE,
    "wrmsse": EvaluationMetric.RMSE,
    "mae": EvaluationMetric.MAE,
    "mape": EvaluationMetric.MAE,
    "smape": EvaluationMetric.MAE,
    "crps": EvaluationMetric.MAE,
    "logloss": EvaluationMetric.LOG_LOSS,
    "nll": EvaluationMetric.LOG_LOSS,
    "bce": EvaluationMetric.LOG_LOSS,
    "auc": EvaluationMetric.AUC,
    "auroc": EvaluationMetric.AUC,
    "roc": EvaluationMetric.AUC,
    "f1": EvaluationMetric.F1,
    "fbeta": EvaluationMetric.F1,
    "dice": EvaluationMetric.F1,
    "iou": EvaluationMetric.F1,
    "miou": EvaluationMetric.F1,
    "mcc": EvaluationMetric.F1,
    "map": EvaluationMetric.MAP,
    "auprc": EvaluationMetric.MAP,
    "ndcg": EvaluationMetric.NDCG,
    "acc": EvaluationMetric.ACCURACY,
}
"""Acronym tokens matched as whole words once phrase matching fails."""


@dataclass(frozen=True, slots=True)
class MetricResolution:
    """Outcome of resolving a free-text metric name.

    @notice: |
        Outcome of resolving a free-text metric name.

    @dev: |
        Carries the resolved metric, its optimisation direction, the
        normalised text used for matching, and whether a table entry matched
        (``False`` means the default metric was substituted).

        @pattern:
            name: value-object
            rationale: "Keeps resolution provenance alongside the resolved metric."
            violations: "Returning a bare metric hides unrecognised platform strings."
    """

    metric: EvaluationMetric
    direction: MetricDirection
    normalized: str
    matched: bool


def normalize_metric_text(raw: str) -> str:
    """Normalise a platform metric string for table matching.

    @notice: |
        Lowercases a metric name and splits camelCase into space-separated words.

    @dev: |
        Inserts a boundary before each uppercase run that follows a lowercase
        character or digit, then collapses every non-alphanumeric run into a
        single space so ``"RootMeanSquaredError"``, ``"Root Mean Squared Error"``
        and ``"root_mean_squared_error"`` all normalise identically.
    """
    spaced = _CAMEL_BOUNDARY.sub(" ", raw)
    return _NON_ALNUM.sub(" ", spaced.lower()).strip()


def resolve_evaluation_metric(raw: str | None) -> MetricResolution:
    """Resolve a free-text metric name to a metric and optimisation direction.

    @notice: |
        Maps a platform metric string onto EvaluationMetric and its direction.

    @dev: |
        Matches ordered multi-word phrases first, then whole-word acronym
        tokens, so specific spellings win over generic ones. Unrecognised text
        falls back to DEFAULT_METRIC and is reported to logfire so missing
        spellings surface instead of silently inverting the search direction.
    """
    normalized = normalize_metric_text(raw or "")
    metric = _match_metric(normalized)
    if metric is None:
        if normalized:
            logfire.warning("metric_unrecognized", raw_metric=raw, normalized=normalized, fallback=DEFAULT_METRIC.value)
        return MetricResolution(
            metric=DEFAULT_METRIC, direction=METRIC_DIRECTIONS[DEFAULT_METRIC], normalized=normalized, matched=False
        )

    return MetricResolution(metric=metric, direction=METRIC_DIRECTIONS[metric], normalized=normalized, matched=True)


def _match_metric(normalized: str) -> EvaluationMetric | None:
    if not normalized:
        return None

    for phrase, metric in _METRIC_PHRASES:
        if phrase in normalized:
            return metric

    for token in normalized.split():
        token_metric = _METRIC_TOKENS.get(token)
        if token_metric is not None:
            return token_metric

    return None
