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


_SCHEMA = CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"])


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


def test_profile_detects_hyphenated_vision_tags() -> None:
    """Tags like ``computer-vision`` and ``object-detection`` map to vision."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.ACCURACY, tags=frozenset({"computer-vision"})), _SCHEMA
    )
    assert profile.problem_type == ProblemType.VISION_CLASSIFICATION

    profile = build_problem_profile(_competition(EvaluationMetric.RMSE, tags=frozenset({"object-detection"})), _SCHEMA)
    assert profile.problem_type == ProblemType.VISION_REGRESSION


def test_profile_detects_image_subdomain_tags() -> None:
    """Image classification / segmentation tags map to vision regardless of separator."""
    for tag in ("image-classification", "image_segmentation", "Image Recognition", "satellite-imagery"):
        profile = build_problem_profile(_competition(EvaluationMetric.ACCURACY, tags=frozenset({tag})), _SCHEMA)
        assert profile.problem_type == ProblemType.VISION_CLASSIFICATION, tag


def test_profile_detects_hyphenated_text_tags() -> None:
    """Tags like ``text-classification`` and ``natural-language-processing`` map to text."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.ACCURACY, tags=frozenset({"text-classification"})), _SCHEMA
    )
    assert profile.problem_type == ProblemType.TEXT_CLASSIFICATION

    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE, tags=frozenset({"natural-language-processing"})), _SCHEMA
    )
    assert profile.problem_type == ProblemType.TEXT_REGRESSION

    for tag in ("sentiment-analysis", "named-entity-recognition", "question-answering", "machine-translation"):
        profile = build_problem_profile(_competition(EvaluationMetric.ACCURACY, tags=frozenset({tag})), _SCHEMA)
        assert profile.problem_type == ProblemType.TEXT_CLASSIFICATION, tag


def test_profile_still_handles_exact_word_tags() -> None:
    """The original single-word tag conventions still resolve to their domains."""
    vision_profile = build_problem_profile(_competition(EvaluationMetric.RMSE, tags=frozenset({"vision"})), _SCHEMA)
    assert vision_profile.problem_type == ProblemType.VISION_REGRESSION

    text_profile = build_problem_profile(_competition(EvaluationMetric.ACCURACY, tags=frozenset({"NLP"})), _SCHEMA)
    assert text_profile.problem_type == ProblemType.TEXT_CLASSIFICATION


def test_profile_does_not_match_substring_within_unrelated_word() -> None:
    """Word-boundary matching prevents false positives like ``context`` matching ``text``."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.ACCURACY, tags=frozenset({"context-aware", "subimagery-coherent"})), _SCHEMA
    )
    assert profile.problem_type == ProblemType.TABULAR_CLASSIFICATION


def test_profile_falls_through_to_tabular_for_unknown_tags() -> None:
    """Unrecognized domain tags still fall through to tabular."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSLE, tags=frozenset({"finance", "energy", "weather"})), _SCHEMA
    )
    assert profile.problem_type == ProblemType.TABULAR_REGRESSION
