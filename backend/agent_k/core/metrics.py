"""Evaluation metric resolution and semantics for AGENT-K.

@notice: |
    Evaluation metric resolution and semantics for AGENT-K.

@dev: |
    Single source of truth for metric direction, classification semantics, and
    probability output requirements. Platform adapters resolve free-text metric
    names here instead of pattern matching inline.

@graph:
    id: agent_k.core.metrics
    provides:
        - agent_k.core.metrics
    pattern: metric-resolution

@agent-guidance:
    do:
        - "Use agent_k.core.metrics as the canonical home for this capability."
        - "Resolve platform metric strings with resolve_evaluation_metric."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Duplicate metric direction or classification sets in other modules."

@human-review:
    last-verified: 2026-10-23
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
    "METRIC_DIRECTIONS",
    "MetricResolution",
    "PROBABILITY_METRICS",
    "RANKING_METRICS",
    "REGRESSION_METRICS",
    "is_classification_metric",
    "metric_direction_for",
    "resolve_evaluation_metric",
    "uses_probability_output",
)

METRIC_DIRECTIONS: Final[dict[EvaluationMetric, MetricDirection]] = {
    EvaluationMetric.ACCURACY: "maximize",
    EvaluationMetric.AUC: "maximize",
    EvaluationMetric.F1: "maximize",
    EvaluationMetric.LOG_LOSS: "minimize",
    EvaluationMetric.RMSE: "minimize",
    EvaluationMetric.MAE: "minimize",
    EvaluationMetric.RMSLE: "minimize",
    EvaluationMetric.R2: "maximize",
    EvaluationMetric.MAP: "maximize",
    EvaluationMetric.NDCG: "maximize",
}
"""Optimisation direction for every supported metric."""

CLASSIFICATION_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {EvaluationMetric.ACCURACY, EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS, EvaluationMetric.F1}
)
"""Metrics whose targets are discrete labels."""

PROBABILITY_METRICS: Final[frozenset[EvaluationMetric]] = frozenset({EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS})
"""Metrics scored on predicted probabilities rather than hard labels."""

RANKING_METRICS: Final[frozenset[EvaluationMetric]] = frozenset({EvaluationMetric.MAP, EvaluationMetric.NDCG})
"""Metrics scored on ranked candidate lists."""

REGRESSION_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {EvaluationMetric.RMSE, EvaluationMetric.MAE, EvaluationMetric.RMSLE, EvaluationMetric.R2}
)
"""Metrics whose targets are continuous values."""

_MINIMIZE_NAME_TOKENS: Final[tuple[str, ...]] = (
    "error",
    "loss",
    "deviation",
    "deviance",
    "distance",
    "divergence",
    "residual",
    "penalty",
    "regret",
)
"""Name fragments that identify an error metric when the metric is unknown."""

_MAXIMIZE_NAME_TOKENS: Final[tuple[str, ...]] = (
    "accuracy",
    "score",
    "gain",
    "precision",
    "recall",
    "kappa",
    "correlation",
    "coefficient",
    "index",
    "similarity",
    "agreement",
)
"""Name fragments that identify a reward metric when the metric is unknown."""

_METRIC_PATTERNS: Final[tuple[tuple[re.Pattern[str], EvaluationMetric], ...]] = (
    # Probability losses first: "multi class logarithmic loss" must not read as RMSLE.
    (re.compile(r"\b(log[\s_-]*loss|logloss|cross[\s_-]*entropy)\b"), EvaluationMetric.LOG_LOSS),
    (re.compile(r"logarithmic[\s_-]+loss"), EvaluationMetric.LOG_LOSS),
    (re.compile(r"\b(rmsle|msle)\b"), EvaluationMetric.RMSLE),
    (re.compile(r"squared[\s_-]+log(arithmic)?[\s_-]+error"), EvaluationMetric.RMSLE),
    (re.compile(r"\b(rmse|mse|rmspe|rmsse|wrmsse|crps)\b"), EvaluationMetric.RMSE),
    (re.compile(r"squared[\s_-]+error|ranked[\s_-]+probability[\s_-]+score"), EvaluationMetric.RMSE),
    (re.compile(r"\b(mae|smape|mape|mad)\b"), EvaluationMetric.MAE),
    (re.compile(r"absolute[\s_-]+(percentage[\s_-]+)?(error|deviation)"), EvaluationMetric.MAE),
    (re.compile(r"\b(auc|roc|auroc|gini)\b"), EvaluationMetric.AUC),
    (re.compile(r"receiver[\s_-]+operating"), EvaluationMetric.AUC),
    (re.compile(r"\bndcg\b|discounted[\s_-]+cumulative[\s_-]+gain"), EvaluationMetric.NDCG),
    (re.compile(r"\bmap\b|\bmap@\d+|mean[\s_-]+average[\s_-]+precision"), EvaluationMetric.MAP),
    (
        re.compile(r"\bf[\s_-]*(1|beta|0\.5)\b|f[\s_-]*score|dice|jaccard|\biou\b|intersection[\s_-]+over"),
        EvaluationMetric.F1,
    ),
    (re.compile(r"\b(r2|r\^2)\b|r[\s_-]*squared|coefficient[\s_-]+of[\s_-]+determination"), EvaluationMetric.R2),
    # Matthews correlation is a label metric, so it must win over the correlation rule below.
    (re.compile(r"\bmcc\b|matthews"), EvaluationMetric.ACCURACY),
    (re.compile(r"\b(accuracy|acc|kappa)\b"), EvaluationMetric.ACCURACY),
    (re.compile(r"\b(pearson|spearman)\b|correlation"), EvaluationMetric.R2),
)
"""Ordered patterns mapping platform metric text onto supported metrics."""

_UNKNOWN_MAXIMIZE_METRIC: Final[EvaluationMetric] = EvaluationMetric.ACCURACY
"""Metric used when an unknown name reads as a reward."""

_UNKNOWN_MINIMIZE_METRIC: Final[EvaluationMetric] = EvaluationMetric.RMSE
"""Metric used when an unknown name reads as an error."""


@dataclass(frozen=True, slots=True)
class MetricResolution:
    """Outcome of resolving a platform metric name.

    @notice: |
        Outcome of resolving a platform metric name.

    @dev: |
        ``matched`` is False when the raw name hit no known pattern and the
        metric was inferred from reward/error wording alone.

        @pattern:
            name: resolution-result
            rationale: "Carries the resolved metric with its provenance."
            violations: "Bare tuples hide whether resolution was a guess."
    """

    metric: EvaluationMetric
    direction: MetricDirection
    raw: str
    matched: bool


def metric_direction_for(metric: EvaluationMetric) -> MetricDirection:
    """Return the optimisation direction for a metric.

    @notice: |
        Returns "maximize" or "minimize" for the given metric.

    @dev: |
        Falls back to "maximize" only for metrics missing from
        METRIC_DIRECTIONS, which the enum coverage test forbids.
    """
    return METRIC_DIRECTIONS.get(metric, "maximize")


def is_classification_metric(metric: EvaluationMetric) -> bool:
    """Report whether a metric scores discrete label predictions.

    @notice: |
        True when the metric expects class labels or class probabilities.

    @dev: |
        Ranking metrics are excluded: they are scored on ordered candidates,
        not on a single label per row.
    """
    return metric in CLASSIFICATION_METRICS


def uses_probability_output(metric: EvaluationMetric) -> bool:
    """Report whether a metric is scored on predicted probabilities.

    @notice: |
        True when submissions must carry probabilities instead of labels.

    @dev: |
        Drives predict_proba selection in generated baseline solutions.
    """
    return metric in PROBABILITY_METRICS


def resolve_evaluation_metric(raw: str | None) -> MetricResolution:
    """Resolve free-text platform metric names onto supported metrics.

    @notice: |
        Maps a platform metric string to an EvaluationMetric and direction.

    @dev: |
        Patterns are ordered so that compound names resolve to the most
        specific metric ("multiclass logarithmic loss" is a log loss, not an
        RMSLE). Unrecognised names fall back to reward/error wording so the
        optimisation direction stays correct, and the miss is logged.
    """
    text = (raw or "").strip()
    if not text:
        return MetricResolution(
            metric=_UNKNOWN_MAXIMIZE_METRIC,
            direction=metric_direction_for(_UNKNOWN_MAXIMIZE_METRIC),
            raw=text,
            matched=False,
        )

    normalized = text.lower()
    for pattern, metric in _METRIC_PATTERNS:
        if pattern.search(normalized):
            return MetricResolution(metric=metric, direction=metric_direction_for(metric), raw=text, matched=True)

    metric = _infer_metric_from_wording(normalized)
    logfire.warn(
        "unresolved evaluation metric",
        raw_metric=text,
        fallback_metric=metric.value,
        fallback_direction=metric_direction_for(metric),
    )
    return MetricResolution(metric=metric, direction=metric_direction_for(metric), raw=text, matched=False)


def _infer_metric_from_wording(normalized: str) -> EvaluationMetric:
    minimize_hit = next((token for token in _MINIMIZE_NAME_TOKENS if token in normalized), None)
    maximize_hit = next((token for token in _MAXIMIZE_NAME_TOKENS if token in normalized), None)
    if minimize_hit is not None and maximize_hit is None:
        return _UNKNOWN_MINIMIZE_METRIC
    return _UNKNOWN_MAXIMIZE_METRIC
