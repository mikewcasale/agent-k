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


def _competition(metric: EvaluationMetric, tags: frozenset[str] = frozenset({"tabular"})) -> Competition:
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


def test_build_problem_profile_time_series_from_schema() -> None:
    """A detected temporal column promotes a tabular competition to time series."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE),
        CompetitionSchema(
            id_column="id", target_columns=["sales"], train_target_columns=["sales"], time_column="sale_date"
        ),
    )
    assert profile.problem_type == ProblemType.TIME_SERIES_REGRESSION
    assert profile.time_column == "sale_date"
    assert profile.is_temporal is True


def test_build_problem_profile_time_series_from_tags() -> None:
    """Temporal competition tags promote the profile even without a detected column."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.AUC, tags=frozenset({"time series"})),
        CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"]),
    )
    assert profile.problem_type == ProblemType.TIME_SERIES_CLASSIFICATION
    assert profile.is_temporal is True


def test_build_problem_profile_tabular_is_not_temporal() -> None:
    """Plain tabular competitions keep the tabular profile and stay non-temporal."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE),
        CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"]),
    )
    assert profile.problem_type == ProblemType.TABULAR_REGRESSION
    assert profile.is_temporal is False


def test_time_series_policies_match_structured_defaults() -> None:
    """Time-series problems reuse the structured (tabular) policy thresholds."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE),
        CompetitionSchema(
            id_column="id", target_columns=["sales"], train_target_columns=["sales"], time_column="sale_date"
        ),
    )
    technique_policy = build_technique_policy(profile)
    fitness_policy = build_fitness_policy(profile, None, max_runtime_ms=1000)

    assert technique_policy.enable_outlier_clipping is True
    assert technique_policy.enable_target_transform is True
    assert fitness_policy.complexity_threshold == 800


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
