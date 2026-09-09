"""Tests for generic strategy utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime

from agent_k.core.data import CompetitionSchema
from agent_k.core.modality import ColumnModality, DataModality, FeatureModality
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.core.strategy import (
    FitnessInput,
    ProblemType,
    TechniquePolicy,
    apply_solution_policy,
    build_fitness_function,
    build_fitness_policy,
    build_problem_profile,
    build_technique_guidance,
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


def _schema() -> CompetitionSchema:
    return CompetitionSchema(id_column="id", target_columns=["target"], train_target_columns=["target"])


def _modality(*entries: tuple[str, FeatureModality, str | None]) -> DataModality:
    return DataModality(
        columns=tuple(
            ColumnModality(
                column=column, modality=modality, distinct_ratio=1.0, mean_token_count=8.0, media_kind=media_kind
            )
            for column, modality, media_kind in entries
        ),
        sampled_rows=len(entries),
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


def test_build_problem_profile_uses_modality_when_tags_are_untagged() -> None:
    """Free-text feature columns select the text family when tags do not decide."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.AUC, tags=frozenset()),
        _schema(),
        _modality(("comment", FeatureModality.TEXT, None)),
    )
    assert profile.problem_type == ProblemType.TEXT_CLASSIFICATION
    assert profile.text_feature_columns == ("comment",)
    assert profile.has_tabular_features is False


def test_build_problem_profile_detects_media_references() -> None:
    """Image file references select the vision family when tags do not decide."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE, tags=frozenset()),
        _schema(),
        _modality(("image_path", FeatureModality.FILE_REFERENCE, "image")),
    )
    assert profile.problem_type == ProblemType.VISION_REGRESSION
    assert profile.file_reference_columns == ("image_path",)


def test_build_problem_profile_keeps_platform_tags_authoritative() -> None:
    """Explicit platform tags win over inferred modality."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.AUC, tags=frozenset({"nlp"})),
        _schema(),
        _modality(("age", FeatureModality.NUMERIC, None)),
    )
    assert profile.problem_type == ProblemType.TEXT_CLASSIFICATION


def test_build_technique_guidance_steers_text_columns_away_from_one_hot() -> None:
    """Text columns get vectoriser guidance and drop tabular-only advice."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.AUC, tags=frozenset()),
        _schema(),
        _modality(("comment", FeatureModality.TEXT, None)),
    )
    guidance = " ".join(build_technique_guidance(profile))
    assert "TfidfVectorizer" in guidance
    assert "get_dummies" not in guidance


def test_build_technique_guidance_flags_undecodable_media_only_features() -> None:
    """Media-only datasets are told to fall back to the training target prior."""
    profile = build_problem_profile(
        _competition(EvaluationMetric.RMSE, tags=frozenset()),
        _schema(),
        _modality(("image_path", FeatureModality.FILE_REFERENCE, "image")),
    )
    guidance = " ".join(build_technique_guidance(profile))
    assert "raw media decoding is unavailable" in guidance
    assert "target prior" in guidance


def test_build_technique_guidance_defaults_to_tabular_advice() -> None:
    """Profiles without modality evidence keep the existing tabular guidance."""
    profile = build_problem_profile(_competition(EvaluationMetric.RMSE), _schema())
    guidance = " ".join(build_technique_guidance(profile))
    assert "KNeighborsRegressor" in guidance
    assert "get_dummies" in guidance
