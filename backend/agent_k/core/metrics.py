"""Evaluation metric registry and parsing for AGENT-K.

@notice: |
    Central registry for evaluation metric parsing, direction, and taxonomy.

@dev: |
    Kaggle's `evaluationMetric` field is free-form text (e.g. "AUC",
    "AreaUnderROCCurve", "SymmetricMeanAbsolutePercentageError"). Downstream
    strategy/evolution code needs a canonical `EvaluationMetric`, a
    minimize/maximize direction, and taxonomy flags. Rather than scattering
    those maps across adapters and nodes, this module owns them.

@graph:
    id: agent_k.core.metrics
    provides:
        - agent_k.core.metrics:parse_metric
        - agent_k.core.metrics:metric_direction
        - agent_k.core.metrics:is_classification_metric
        - agent_k.core.metrics:uses_probability
        - agent_k.core.metrics:CLASSIFICATION_METRICS
        - agent_k.core.metrics:PROBA_METRICS
        - agent_k.core.metrics:REGRESSION_METRICS
    pattern: registry

@similar:
    - id: agent_k.core.models
        when: "EvaluationMetric enum definition; this module maps to it."
    - id: agent_k.adapters.kaggle
        when: "Adapter parsing of Kaggle competition payloads."

@agent-guidance:
    do:
        - "Add new metric aliases here; keep adapters call `parse_metric`."
    do_not:
        - "Duplicate metric-name → enum tables in adapters or nodes."

@human-review:
    last-verified: 2026-08-07
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re
from typing import TYPE_CHECKING, Annotated, Final

import logfire

from agent_k.core.models import EvaluationMetric
from agent_k.core.sage import Doc

if TYPE_CHECKING:
    from agent_k.core.types import MetricDirection

__all__ = (
    "CLASSIFICATION_METRICS",
    "PROBA_METRICS",
    "RANKING_METRICS",
    "REGRESSION_METRICS",
    "is_classification_metric",
    "is_ranking_metric",
    "is_regression_metric",
    "metric_direction",
    "parse_metric",
    "uses_probability",
)

SCHEMA_VERSION: Final[str] = "1.0.0"

CLASSIFICATION_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {
        EvaluationMetric.ACCURACY,
        EvaluationMetric.AUC,
        EvaluationMetric.LOG_LOSS,
        EvaluationMetric.MULTI_LOG_LOSS,
        EvaluationMetric.F1,
        EvaluationMetric.BALANCED_ACCURACY,
        EvaluationMetric.MCC,
        EvaluationMetric.QUADRATIC_KAPPA,
    }
)
"""Metrics that indicate a classification task."""

REGRESSION_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {
        EvaluationMetric.RMSE,
        EvaluationMetric.MAE,
        EvaluationMetric.RMSLE,
        EvaluationMetric.MEDAE,
        EvaluationMetric.R2,
        EvaluationMetric.SMAPE,
        EvaluationMetric.MAPE,
        EvaluationMetric.MCRMSE,
        EvaluationMetric.SPEARMAN,
        EvaluationMetric.PEARSON,
    }
)
"""Metrics that indicate a regression / continuous-target task."""

RANKING_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {EvaluationMetric.MAP, EvaluationMetric.NDCG, EvaluationMetric.MRR}
)
"""Metrics that indicate a ranking task."""

PROBA_METRICS: Final[frozenset[EvaluationMetric]] = frozenset(
    {EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS, EvaluationMetric.MULTI_LOG_LOSS}
)
"""Metrics that require probability outputs, not hard labels."""

_METRIC_DIRECTIONS: Final[dict[EvaluationMetric, str]] = {
    EvaluationMetric.ACCURACY: "maximize",
    EvaluationMetric.AUC: "maximize",
    EvaluationMetric.F1: "maximize",
    EvaluationMetric.BALANCED_ACCURACY: "maximize",
    EvaluationMetric.MCC: "maximize",
    EvaluationMetric.QUADRATIC_KAPPA: "maximize",
    EvaluationMetric.R2: "maximize",
    EvaluationMetric.SPEARMAN: "maximize",
    EvaluationMetric.PEARSON: "maximize",
    EvaluationMetric.MAP: "maximize",
    EvaluationMetric.NDCG: "maximize",
    EvaluationMetric.MRR: "maximize",
    EvaluationMetric.LOG_LOSS: "minimize",
    EvaluationMetric.MULTI_LOG_LOSS: "minimize",
    EvaluationMetric.RMSE: "minimize",
    EvaluationMetric.MAE: "minimize",
    EvaluationMetric.RMSLE: "minimize",
    EvaluationMetric.MEDAE: "minimize",
    EvaluationMetric.SMAPE: "minimize",
    EvaluationMetric.MAPE: "minimize",
    EvaluationMetric.MCRMSE: "minimize",
}

_METRIC_ALIASES: Final[dict[str, EvaluationMetric]] = {
    # Classification
    "accuracy": EvaluationMetric.ACCURACY,
    "categorizationaccuracy": EvaluationMetric.ACCURACY,
    "acc": EvaluationMetric.ACCURACY,
    "auc": EvaluationMetric.AUC,
    "aucroc": EvaluationMetric.AUC,
    "rocauc": EvaluationMetric.AUC,
    "areaunderroc": EvaluationMetric.AUC,
    "areaundercurve": EvaluationMetric.AUC,
    "areaunderroccurve": EvaluationMetric.AUC,
    "logloss": EvaluationMetric.LOG_LOSS,
    "binarylogloss": EvaluationMetric.LOG_LOSS,
    "binarycrossentropy": EvaluationMetric.LOG_LOSS,
    "multilogloss": EvaluationMetric.MULTI_LOG_LOSS,
    "multiclasslogloss": EvaluationMetric.MULTI_LOG_LOSS,
    "categoricalcrossentropy": EvaluationMetric.MULTI_LOG_LOSS,
    "f1": EvaluationMetric.F1,
    "f1score": EvaluationMetric.F1,
    "macrof1": EvaluationMetric.F1,
    "microf1": EvaluationMetric.F1,
    "weightedf1": EvaluationMetric.F1,
    "meanfscore": EvaluationMetric.F1,
    "balancedaccuracy": EvaluationMetric.BALANCED_ACCURACY,
    "mcc": EvaluationMetric.MCC,
    "matthewscorrcoef": EvaluationMetric.MCC,
    "matthewscorrelation": EvaluationMetric.MCC,
    "matthewscorrelationcoefficient": EvaluationMetric.MCC,
    "quadratickappa": EvaluationMetric.QUADRATIC_KAPPA,
    "qwk": EvaluationMetric.QUADRATIC_KAPPA,
    "quadraticweightedkappa": EvaluationMetric.QUADRATIC_KAPPA,
    "cohenkappa": EvaluationMetric.QUADRATIC_KAPPA,
    "cohenskappa": EvaluationMetric.QUADRATIC_KAPPA,
    # Regression
    "rmse": EvaluationMetric.RMSE,
    "rootmeansquarederror": EvaluationMetric.RMSE,
    "meansquarederror": EvaluationMetric.RMSE,
    "mse": EvaluationMetric.RMSE,
    "mae": EvaluationMetric.MAE,
    "meanabsoluteerror": EvaluationMetric.MAE,
    "rmsle": EvaluationMetric.RMSLE,
    "rootmeansquaredlogarithmicerror": EvaluationMetric.RMSLE,
    "rootmeansquaredloge": EvaluationMetric.RMSLE,
    "medae": EvaluationMetric.MEDAE,
    "medianabsoluteerror": EvaluationMetric.MEDAE,
    "r2": EvaluationMetric.R2,
    "rsquared": EvaluationMetric.R2,
    "coefficientofdetermination": EvaluationMetric.R2,
    "smape": EvaluationMetric.SMAPE,
    "symmetricmeanabsolutepercentageerror": EvaluationMetric.SMAPE,
    "mape": EvaluationMetric.MAPE,
    "meanabsolutepercentageerror": EvaluationMetric.MAPE,
    "mcrmse": EvaluationMetric.MCRMSE,
    "meancolumnwiserootmeansquarederror": EvaluationMetric.MCRMSE,
    "meancolumnwiserootmeansquaredlogarithmicerror": EvaluationMetric.MCRMSE,
    "spearman": EvaluationMetric.SPEARMAN,
    "spearmancorrelation": EvaluationMetric.SPEARMAN,
    "spearmanrankcorrelation": EvaluationMetric.SPEARMAN,
    "pearson": EvaluationMetric.PEARSON,
    "pearsoncorrelation": EvaluationMetric.PEARSON,
    # Ranking
    "map": EvaluationMetric.MAP,
    "meanaverageprecision": EvaluationMetric.MAP,
    "ndcg": EvaluationMetric.NDCG,
    "normalizeddiscountedcumulativegain": EvaluationMetric.NDCG,
    "mrr": EvaluationMetric.MRR,
    "meanreciprocalrank": EvaluationMetric.MRR,
}

_METRIC_PATTERNS: Final[tuple[tuple[re.Pattern[str], EvaluationMetric], ...]] = (
    (re.compile(r"quadratic.*kappa|kappa.*quadratic|qwk"), EvaluationMetric.QUADRATIC_KAPPA),
    (re.compile(r"cohen.*kappa|kappa.*cohen"), EvaluationMetric.QUADRATIC_KAPPA),
    (re.compile(r"matthews.*correlation|mcc\b"), EvaluationMetric.MCC),
    (re.compile(r"balanced.*accuracy"), EvaluationMetric.BALANCED_ACCURACY),
    (re.compile(r"multi.*(log.*loss|cross.*entropy)"), EvaluationMetric.MULTI_LOG_LOSS),
    (re.compile(r"(binary|log|logistic).*(log.*loss|cross.*entropy)"), EvaluationMetric.LOG_LOSS),
    (re.compile(r"logloss|log.loss|log\s?loss"), EvaluationMetric.LOG_LOSS),
    (re.compile(r"area.*under.*(roc|curve)"), EvaluationMetric.AUC),
    (re.compile(r"symmetric.*mean.*absolute.*percentage|smape"), EvaluationMetric.SMAPE),
    (re.compile(r"mean.*absolute.*percentage|mape"), EvaluationMetric.MAPE),
    (re.compile(r"median.*absolute.*error|medae"), EvaluationMetric.MEDAE),
    (re.compile(r"mean.*column.*wise.*(log|logarithmic)"), EvaluationMetric.MCRMSE),
    (re.compile(r"mean.*column.*wise.*root.*mean.*squared"), EvaluationMetric.MCRMSE),
    (re.compile(r"mcrmse"), EvaluationMetric.MCRMSE),
    (re.compile(r"spearman"), EvaluationMetric.SPEARMAN),
    (re.compile(r"pearson"), EvaluationMetric.PEARSON),
    (re.compile(r"coefficient.*of.*determination|r.?squared|\br2\b"), EvaluationMetric.R2),
    (re.compile(r"root.*mean.*squared.*(log|logarithmic)|rmsle"), EvaluationMetric.RMSLE),
    (re.compile(r"root.*mean.*squared.*error|\brmse\b"), EvaluationMetric.RMSE),
    (re.compile(r"mean.*squared.*error|\bmse\b"), EvaluationMetric.RMSE),
    (re.compile(r"mean.*absolute.*error|\bmae\b"), EvaluationMetric.MAE),
    (re.compile(r"mean.*reciprocal.*rank|\bmrr\b"), EvaluationMetric.MRR),
    (re.compile(r"mean.*average.*precision|\bmap\b|map@"), EvaluationMetric.MAP),
    (re.compile(r"normalized.*discounted.*cumulative|\bndcg\b|ndcg@"), EvaluationMetric.NDCG),
    (re.compile(r"(macro|micro|weighted|mean).*f.?score|f.?score|\bf1\b"), EvaluationMetric.F1),
    (re.compile(r"\bauc\b|roc"), EvaluationMetric.AUC),
    (re.compile(r"categorization.*accuracy|classification.*accuracy|accuracy"), EvaluationMetric.ACCURACY),
)


def _normalize(raw: str) -> str:
    """Normalize a raw metric string for alias lookup."""
    return re.sub(r"[^a-z0-9]", "", raw.lower())


def parse_metric(
    raw: Annotated[str | None, Doc("Raw metric string from the platform response.")],
    *,
    default: Annotated[EvaluationMetric, Doc("Fallback metric when the string cannot be resolved.")] = (
        EvaluationMetric.ACCURACY
    ),
) -> tuple[EvaluationMetric, MetricDirection]:
    """Parse a raw metric string into an EvaluationMetric and its direction.

    @notice: |
        Recognizes both compact aliases ("rmse", "auc") and long names
        ("RootMeanSquaredError", "QuadraticWeightedKappa").

    @dev: |
        Lookup order:
          1. Compact alias table (normalized, punctuation stripped).
          2. Substring pattern list (handles verbose Kaggle names).
          3. `default`, with a `logfire` warning tagged `metric_unresolved`.
    """
    if not raw:
        return default, direction_for(default)

    stripped = raw.strip()
    normalized = _normalize(stripped)
    if normalized in _METRIC_ALIASES:
        metric = _METRIC_ALIASES[normalized]
        return metric, direction_for(metric)

    lowered = stripped.lower()
    for pattern, metric in _METRIC_PATTERNS:
        if pattern.search(lowered):
            return metric, direction_for(metric)

    logfire.warning("metric_unresolved", raw=stripped, fallback=default.value)
    return default, direction_for(default)


def direction_for(metric: Annotated[EvaluationMetric, Doc("Metric to look up direction for.")]) -> MetricDirection:
    """Return the canonical optimization direction for a metric."""
    from agent_k.core.types import MetricDirection  # noqa: F401  (needed for type narrowing)

    value = _METRIC_DIRECTIONS.get(metric, "maximize")
    # Guard against typos in the direction table.
    if value not in {"maximize", "minimize"}:
        return "maximize"
    return value  # type: ignore[return-value]


def metric_direction(metric: Annotated[EvaluationMetric, Doc("Metric to look up direction for.")]) -> MetricDirection:
    """Public alias for :func:`direction_for`."""
    return direction_for(metric)


def is_classification_metric(metric: Annotated[EvaluationMetric, Doc("Metric to classify.")]) -> bool:
    """Return True when the metric implies a classification task."""
    return metric in CLASSIFICATION_METRICS


def is_regression_metric(metric: Annotated[EvaluationMetric, Doc("Metric to classify.")]) -> bool:
    """Return True when the metric implies a regression task."""
    return metric in REGRESSION_METRICS


def is_ranking_metric(metric: Annotated[EvaluationMetric, Doc("Metric to classify.")]) -> bool:
    """Return True when the metric implies a ranking task."""
    return metric in RANKING_METRICS


def uses_probability(metric: Annotated[EvaluationMetric, Doc("Metric to check.")]) -> bool:
    """Return True when the metric requires probability-valued predictions."""
    return metric in PROBA_METRICS
