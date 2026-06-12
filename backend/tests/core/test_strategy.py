"""Tests for generic strategy utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime

from agent_k.core.data import CompetitionSchema
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.core.strategy import (
    FitnessInput,
    ProblemType,
    TechniquePolicy,
    apply_solution_policy,
    build_fitness_function,
    build_fitness_policy,
    build_problem_profile,
    build_technique_policy,
)


def _competition(metric: EvaluationMetric, *, tags: frozenset[str] = frozenset({"tabular"})) -> Competition:
    return Competition(
        id="sample-competition",
        title="Sample Competition",
        description=None,
        competition_type=CompetitionType.FEATURED,
        metric=metric,
        metric_direction="minimize",
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
        prize_pool=None,
        max_team_size=1,
        max_daily_submissions=5,
        tags=tags,
        url=None,
    )


def _schema() -> CompetitionSchema:
    return CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"])


def test_build_problem_profile_regression() -> None:
    """Ensure regression competitions map to tabular regression profiles."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSLE),
        CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"]),
    )
    assert profile.problem_type == ProblemType.TABULAR_REGRESSION
    assert profile.is_classification is False


def test_build_problem_profile_classification() -> None:
    """Ensure classification competitions map to tabular classification profiles."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.AUC),
        CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"]),
    )
    assert profile.problem_type == ProblemType.TABULAR_CLASSIFICATION
    assert profile.is_classification is True


def test_fitness_factory_penalizes_runtime_and_complexity() -> None:
    """Penalize fitness when runtime or complexity exceeds thresholds."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE),
        CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"]),
    )
    policy = build_fitness_policy(profile, None, max_runtime_ms=1000, complexity_threshold=10)
    fitness_fn = build_fitness_function(policy)

    base = FitnessInput(cv_score=0.5, runtime_ms=500, complexity=5, valid=True, stage="full", code="print('ok')")
    penalized = FitnessInput(cv_score=0.5, runtime_ms=1500, complexity=20, valid=True, stage="full", code="print('ok')")

    assert fitness_fn(penalized) < fitness_fn(base)


def test_apply_solution_policy_is_noop() -> None:
    """Policy injection is disabled; apply_solution_policy returns code unchanged."""
    code = (
        "import pandas as pd\n"
        "train = pd.read_csv('train.csv')\n"
        "test = pd.read_csv('test.csv')\n"
        "USES_LOG_TARGET = False\n"
    )
    policy = TechniquePolicy(problem_type=ProblemType.TABULAR_REGRESSION, enable_target_transform=True)
    updated, notes = apply_solution_policy(code, policy)
    assert not notes
    # Policy injection is now disabled - code should be returned unchanged
    assert updated == code

    updated_again, notes_again = apply_solution_policy(updated, policy)
    assert updated_again == updated
    assert not notes_again


def test_build_problem_profile_time_series_regression() -> None:
    """Time-series tags should map to time-series regression for regression metrics."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE, tags=frozenset({"time series", "forecasting"})), _schema()
    )
    assert profile.problem_type == ProblemType.TIME_SERIES_REGRESSION
    assert profile.is_classification is False


def test_build_problem_profile_time_series_classification() -> None:
    """Time-series tags should map to time-series classification for classification metrics."""
    profile = build_problem_profile(_competition(EvaluationMetric.AUC, tags=frozenset({"timeseries"})), _schema())
    assert profile.problem_type == ProblemType.TIME_SERIES_CLASSIFICATION
    assert profile.is_classification is True


def test_build_problem_profile_time_series_alternative_tags() -> None:
    """All time-series synonyms in the tag set should be detected."""
    for tag in ("time series", "timeseries", "time-series", "forecasting", "temporal", "seasonal"):
        profile = build_problem_profile(_competition(EvaluationMetric.RMSE, tags=frozenset({tag})), _schema())
        assert profile.problem_type == ProblemType.TIME_SERIES_REGRESSION, (
            f"tag {tag!r} did not map to time-series regression"
        )


def test_build_problem_profile_vision_outranks_time_series() -> None:
    """Vision tags take precedence over time-series tags when both appear."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.AUC, tags=frozenset({"vision", "time series"})), _schema()
    )
    assert profile.problem_type == ProblemType.VISION_CLASSIFICATION


def test_build_problem_profile_text_outranks_time_series() -> None:
    """Text tags take precedence over time-series tags when both appear."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.ACCURACY, tags=frozenset({"nlp", "forecasting"})), _schema()
    )
    assert profile.problem_type == ProblemType.TEXT_CLASSIFICATION


def test_build_technique_policy_time_series_disables_tabular_transforms() -> None:
    """Time-series profiles must not enable tabular outlier clipping or target transform.

    Time-series targets often contain legitimate spikes that quantile clipping would erase,
    and log1p target transforms make poor defaults outside i.i.d. tabular regression.
    """
    profile = build_problem_profile(_competition(EvaluationMetric.RMSE, tags=frozenset({"time series"})), _schema())
    policy = build_technique_policy(profile)
    assert policy.problem_type == ProblemType.TIME_SERIES_REGRESSION
    assert policy.enable_target_transform is False
    assert policy.enable_outlier_clipping is False


def test_build_fitness_policy_time_series_uses_modality_complexity_budget() -> None:
    """Time-series profiles share the non-tabular (lower) complexity threshold."""
    ts_profile = build_problem_profile(_competition(EvaluationMetric.AUC, tags=frozenset({"time series"})), _schema())
    tabular_profile = build_problem_profile(_competition(EvaluationMetric.AUC, tags=frozenset({"tabular"})), _schema())
    ts_policy = build_fitness_policy(ts_profile, None, max_runtime_ms=None)
    tabular_policy = build_fitness_policy(tabular_profile, None, max_runtime_ms=None)
    assert ts_policy.complexity_threshold == 600
    assert tabular_policy.complexity_threshold == 800
