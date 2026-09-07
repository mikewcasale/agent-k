"""Evaluation metric taxonomy and platform metric-name resolution for AGENT-K.

@notice: |
    Evaluation metric taxonomy and platform metric-name resolution for AGENT-K.

@dev: |
    Kaggle reports ``evaluationMetric`` as a free-form identifier that is
    usually CamelCase and unpunctuated (``RootMeanSquaredError``,
    ``QuadraticWeightedKappa``). Substring checks against spaced phrases miss
    those, so this module normalises the raw name before matching and falls
    back to lexical direction/family detection for metrics outside the
    supported taxonomy.

@graph:
    id: agent_k.core.metrics
    provides:
        - agent_k.core.metrics
    pattern: taxonomy

@agent-guidance:
    do:
        - "Use agent_k.core.metrics as the canonical home for this capability."
        - "Resolve platform metric strings with resolve_metric instead of ad-hoc substring checks."
        - "Read classification/probability behaviour from METRIC_SPECS instead of inline metric sets."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Add competition-specific metric handling; rules must stay generic per ML problem family."

@human-review:
    last-verified: 2026-09-07
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re
from dataclasses import dataclass
from typing import Final

from .models import EvaluationMetric
from .types import MetricDirection

__all__ = (
    "METRIC_SPECS",
    "MetricSpec",
    "ResolvedMetric",
    "describe_metric",
    "is_classification_metric",
    "metric_spec",
    "normalize_metric_name",
    "resolve_metric",
    "uses_probability",
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Behavioural profile of a supported evaluation metric.

    @notice: |
        Behavioural profile of a supported evaluation metric.

    @dev: |
        Single source of truth for optimisation direction and prediction shape.
        Consumers read these flags instead of re-declaring inline metric sets.

        @pattern:
            name: taxonomy-entry
            rationale: "Bundles direction and prediction shape per metric."
            violations: "Duplicated metric sets drift between modules."
    """

    direction: MetricDirection
    is_classification: bool
    uses_proba: bool


@dataclass(frozen=True, slots=True)
class ResolvedMetric:
    """Outcome of resolving a platform-reported metric name.

    @notice: |
        Outcome of resolving a platform-reported metric name.

    @dev: |
        ``recognized`` is False when the raw name matched no taxonomy rule and
        the metric/direction were inferred lexically; callers may log that.

        @pattern:
            name: resolution-result
            rationale: "Keeps provenance of the metric decision inspectable."
            violations: "Silent fallbacks hide wrong optimisation directions."
    """

    metric: EvaluationMetric
    direction: MetricDirection
    raw: str
    recognized: bool


METRIC_SPECS: Final[dict[EvaluationMetric, MetricSpec]] = {
    EvaluationMetric.ACCURACY: MetricSpec("maximize", is_classification=True, uses_proba=False),
    EvaluationMetric.AUC: MetricSpec("maximize", is_classification=True, uses_proba=True),
    EvaluationMetric.LOG_LOSS: MetricSpec("minimize", is_classification=True, uses_proba=True),
    EvaluationMetric.F1: MetricSpec("maximize", is_classification=True, uses_proba=False),
    EvaluationMetric.RMSE: MetricSpec("minimize", is_classification=False, uses_proba=False),
    EvaluationMetric.MAE: MetricSpec("minimize", is_classification=False, uses_proba=False),
    EvaluationMetric.RMSLE: MetricSpec("minimize", is_classification=False, uses_proba=False),
    EvaluationMetric.R2: MetricSpec("maximize", is_classification=False, uses_proba=False),
    EvaluationMetric.MAP: MetricSpec("maximize", is_classification=False, uses_proba=False),
    EvaluationMetric.NDCG: MetricSpec("maximize", is_classification=False, uses_proba=False),
}
"""Direction and prediction shape for every supported metric."""

_DEFAULT_METRIC: Final[EvaluationMetric] = EvaluationMetric.ACCURACY

_CAMEL_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

_METRIC_RULES: Final[tuple[tuple[re.Pattern[str], EvaluationMetric], ...]] = (
    (re.compile(r"\brmsle\b|mean squared log|logarithmic error"), EvaluationMetric.RMSLE),
    (re.compile(r"\blog ?loss\b|\bmlogloss\b|cross entropy|multi ?class loss|deviance"), EvaluationMetric.LOG_LOSS),
    (re.compile(r"\brmse\b|\bmse\b|root mean square|squared error"), EvaluationMetric.RMSE),
    (
        re.compile(
            r"\bmae\b|\bmape\b|\bsmape\b|\bwmae\b|\bcrps\b|absolute error|absolute deviation"
            r"|absolute percentage error|ranked probability|pinball|quantile loss"
        ),
        EvaluationMetric.MAE,
    ),
    (re.compile(r"\bauc\b|\bauroc\b|\bau roc\b|\broc\b|\bgini\b|area under"), EvaluationMetric.AUC),
    (re.compile(r"\bf1\b|\bf ?1 score\b|\bfbeta\b|f score|\bdice\b|\bjaccard\b"), EvaluationMetric.F1),
    (re.compile(r"accuracy|\bkappa\b|matthews|\bmcc\b"), EvaluationMetric.ACCURACY),
    (re.compile(r"\bmap\b|mean average precision"), EvaluationMetric.MAP),
    (re.compile(r"\bndcg\b|discounted cumulative gain"), EvaluationMetric.NDCG),
    (re.compile(r"\br ?2\b|\br squared\b|coefficient of determination|explained variance"), EvaluationMetric.R2),
)
"""Ordered rules matched against the normalised metric name; first match wins."""

_MINIMIZE_HINTS: Final[tuple[str, ...]] = (
    "error",
    "loss",
    "deviation",
    "distance",
    "divergence",
    "entropy",
    "perplexity",
    "residual",
    "penalty",
    "cost",
)

_CLASSIFICATION_HINTS: Final[tuple[str, ...]] = (
    "accuracy",
    "error rate",
    "misclassification",
    "auc",
    "roc",
    "kappa",
    "precision",
    "recall",
    "class",
    "label",
    "entropy",
    "matthews",
    "logloss",
    "log loss",
)


def normalize_metric_name(raw: str) -> str:
    """Normalise a platform metric name into lowercase space-separated words.

    @notice: |
        Splits CamelCase metric identifiers and collapses punctuation.

    @dev: |
        ``RootMeanSquaredError`` becomes ``root mean squared error`` and
        ``MAP@3`` becomes ``map 3``, so word-boundary rules can match names
        reported by the Kaggle API as unpunctuated identifiers.
    """
    spaced = _ACRONYM_BOUNDARY.sub(" ", _CAMEL_BOUNDARY.sub(" ", raw))
    return _NON_ALNUM.sub(" ", spaced.lower()).strip()


def metric_spec(metric: EvaluationMetric) -> MetricSpec:
    """Return the behavioural spec for a metric.

    @notice: |
        Looks up direction and prediction shape for a supported metric.

    @dev: |
        Unknown enum members fall back to the accuracy spec so callers never
        need a None branch; every member of EvaluationMetric is covered today.
    """
    return METRIC_SPECS.get(metric, METRIC_SPECS[_DEFAULT_METRIC])


def is_classification_metric(metric: EvaluationMetric) -> bool:
    """Whether the metric scores a classification task.

    @notice: |
        Reports whether a metric belongs to the classification family.

    @dev: |
        Drives estimator selection in prototype generation and profiling.
    """
    return metric_spec(metric).is_classification


def uses_probability(metric: EvaluationMetric) -> bool:
    """Whether the metric scores probability estimates rather than labels.

    @notice: |
        Reports whether submissions must contain probabilities.

    @dev: |
        AUC and log loss require predict_proba output; hard labels score badly.
    """
    return metric_spec(metric).uses_proba


def resolve_metric(raw: str | None) -> ResolvedMetric:
    """Resolve a platform metric name into a supported metric and direction.

    @notice: |
        Maps a free-form platform metric name onto the supported taxonomy.

    @dev: |
        Matches normalised text against ordered taxonomy rules. Unmatched names
        are assigned a proxy metric from lexical direction and family cues so
        the optimisation direction stays correct even for metrics outside the
        taxonomy; ``recognized`` is False in that case.
    """
    text = normalize_metric_name(raw or "")
    if not text:
        return ResolvedMetric(_DEFAULT_METRIC, metric_spec(_DEFAULT_METRIC).direction, raw or "", recognized=False)

    for pattern, metric in _METRIC_RULES:
        if pattern.search(text):
            return ResolvedMetric(metric, metric_spec(metric).direction, raw or "", recognized=True)

    metric = _proxy_metric(text)
    return ResolvedMetric(metric, metric_spec(metric).direction, raw or "", recognized=False)


def describe_metric(metric: EvaluationMetric, direction: MetricDirection, metric_name: str | None = None) -> str:
    """Render a metric for agent prompts, preserving the platform name.

    @notice: |
        Formats metric, direction, and the platform-reported name for prompts.

    @dev: |
        The platform name is appended only when it carries information the enum
        value does not, so agents can optimise metrics the taxonomy approximates.
    """
    rendered = f"{metric.value} ({direction})"
    if not metric_name:
        return rendered
    normalized = normalize_metric_name(metric_name).replace(" ", "")
    if normalized == metric.value.lower():
        return rendered
    return f"{rendered}, reported by the platform as '{metric_name}'"


def _proxy_metric(text: str) -> EvaluationMetric:
    """Pick the closest supported metric for an unrecognised metric name."""
    minimize = any(hint in text for hint in _MINIMIZE_HINTS)
    classification = any(hint in text for hint in _CLASSIFICATION_HINTS)
    if classification:
        return EvaluationMetric.LOG_LOSS if minimize else EvaluationMetric.ACCURACY
    return EvaluationMetric.RMSE if minimize else EvaluationMetric.R2
