"""Tests for core domain models.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_k.core.models import (
    Competition,
    CompetitionType,
    ErrorEvent,
    EvaluationMetric,
    EvolutionState,
    GenerationMetrics,
    LeaderboardEntry,
    MemoryEntry,
    MemoryState,
    MissionCriteria,
    PlannedTask,
    metric_direction_for,
)
from agent_k.mission.state import MissionResult

__all__ = ()


class TestCompetitionType:
    """Tests for the CompetitionType enum."""

    def test_featured_value(self) -> None:
        """Featured type should have correct value."""
        assert CompetitionType.FEATURED.value == "featured"

    def test_research_value(self) -> None:
        """Research type should have correct value."""
        assert CompetitionType.RESEARCH.value == "research"

    def test_getting_started_value(self) -> None:
        """Getting started type should have correct value."""
        assert CompetitionType.GETTING_STARTED.value == "getting_started"


class TestEvaluationMetric:
    """Tests for the EvaluationMetric enum."""

    def test_accuracy_value(self) -> None:
        """Accuracy metric should have correct value."""
        assert EvaluationMetric.ACCURACY.value == "accuracy"

    def test_auc_value(self) -> None:
        """AUC metric should have correct value."""
        assert EvaluationMetric.AUC.value == "auc"

    def test_rmse_value(self) -> None:
        """RMSE metric should have correct value."""
        assert EvaluationMetric.RMSE.value == "rmse"


class TestCompetition:
    """Tests for the Competition model."""

    def test_minimal_creation(self) -> None:
        """Competition should be created with minimal fields."""
        comp = Competition(
            id="titanic",
            title="Titanic",
            competition_type=CompetitionType.GETTING_STARTED,
            metric=EvaluationMetric.ACCURACY,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
        )

        assert comp.id == "titanic"
        assert comp.title == "Titanic"
        assert comp.competition_type == CompetitionType.GETTING_STARTED

    def test_is_frozen(self) -> None:
        """Competition should be immutable."""
        comp = Competition(
            id="titanic",
            title="Titanic",
            competition_type=CompetitionType.GETTING_STARTED,
            metric=EvaluationMetric.ACCURACY,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
        )

        with pytest.raises(ValidationError):
            comp.id = "new_id"  # type: ignore

    def test_invalid_id_pattern(self) -> None:
        """Competition should reject invalid ID patterns."""
        with pytest.raises(ValidationError):
            Competition(
                id="Invalid ID!",  # Contains invalid characters
                title="Test",
                competition_type=CompetitionType.FEATURED,
                metric=EvaluationMetric.ACCURACY,
                deadline=datetime(2030, 1, 1, tzinfo=UTC),
            )


class TestCompetitionPayloadCoercion:
    """Tests for the field shapes platforms and language models emit."""

    @staticmethod
    def _payload(**overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": "house-prices-advanced-regression-techniques",
            "title": "House Prices - Advanced Regression Techniques",
            "competition_type": "Getting Started",
            "metric": "RMSE",
            "deadline": "2030-01-01T23:59:00",
        }
        payload.update(overrides)
        return payload

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("RMSE", EvaluationMetric.RMSE),
            ("rmsle", EvaluationMetric.RMSLE),
            ("log_loss", EvaluationMetric.LOG_LOSS),
            ("LOG_LOSS", EvaluationMetric.LOG_LOSS),
            ("Log Loss", EvaluationMetric.LOG_LOSS),
            ("Accuracy", EvaluationMetric.ACCURACY),
            ("NDCG", EvaluationMetric.NDCG),
        ],
    )
    def test_metric_label_casing(self, raw: str, expected: EvaluationMetric) -> None:
        """Metric labels should resolve regardless of casing or separators."""
        comp = Competition.model_validate(self._payload(metric=raw))

        assert comp.metric is expected

    def test_unknown_metric_is_still_rejected(self) -> None:
        """Labels outside the taxonomy should not be silently remapped."""
        with pytest.raises(ValidationError):
            Competition.model_validate(self._payload(metric="quadratic weighted kappa"))

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Getting Started", CompetitionType.GETTING_STARTED),
            ("getting-started", CompetitionType.GETTING_STARTED),
            ("GETTING_STARTED", CompetitionType.GETTING_STARTED),
            ("Featured", CompetitionType.FEATURED),
        ],
    )
    def test_competition_type_label_casing(self, raw: str, expected: CompetitionType) -> None:
        """Category labels should resolve regardless of casing or separators."""
        comp = Competition.model_validate(self._payload(competition_type=raw))

        assert comp.competition_type is expected

    @pytest.mark.parametrize("metric", list(EvaluationMetric))
    def test_direction_defaults_from_metric(self, metric: EvaluationMetric) -> None:
        """An omitted direction should follow the metric, not the field default."""
        comp = Competition.model_validate(self._payload(metric=metric))

        assert comp.metric_direction == metric_direction_for(metric)

    def test_error_metric_does_not_default_to_maximize(self) -> None:
        """Error metrics must not inherit the maximize field default."""
        comp = Competition.model_validate(self._payload(metric=EvaluationMetric.RMSE))

        assert comp.metric_direction == "minimize"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("minimize", "minimize"),
            ("Minimise", "minimize"),
            ("min", "minimize"),
            ("lower is better", "minimize"),
            ("maximize", "maximize"),
            ("higher is better", "maximize"),
        ],
    )
    def test_direction_phrasing(self, raw: str, expected: str) -> None:
        """Direction phrasing should resolve to the canonical literal."""
        comp = Competition.model_validate(self._payload(metric_direction=raw))

        assert comp.metric_direction == expected

    def test_explicit_direction_overrides_metric_default(self) -> None:
        """A platform-supplied direction should win over the metric default."""
        comp = Competition.model_validate(self._payload(metric=EvaluationMetric.RMSE, metric_direction="maximize"))

        assert comp.metric_direction == "maximize"

    def test_uninterpretable_direction_falls_back_to_metric(self) -> None:
        """A direction that carries no ordering should fall back to the metric."""
        comp = Competition.model_validate(self._payload(metric=EvaluationMetric.AUC, metric_direction="best"))

        assert comp.metric_direction == "maximize"

    def test_naive_deadline_string_is_utc(self) -> None:
        """A deadline without an offset should be read as UTC, not rejected."""
        comp = Competition.model_validate(self._payload(deadline="2030-01-01T23:59:00"))

        assert comp.deadline == datetime(2030, 1, 1, 23, 59, tzinfo=UTC)

    def test_naive_deadline_datetime_is_utc(self) -> None:
        """A naive datetime should be read as UTC, not rejected."""
        comp = Competition.model_validate(self._payload(deadline=datetime(2030, 1, 1, 23, 59)))

        assert comp.deadline == datetime(2030, 1, 1, 23, 59, tzinfo=UTC)

    def test_aware_deadline_is_preserved(self) -> None:
        """An offset-carrying deadline should keep its instant."""
        comp = Competition.model_validate(self._payload(deadline="2030-01-01T23:59:00+02:00"))

        assert comp.deadline == datetime(2030, 1, 1, 21, 59, tzinfo=UTC)

    def test_delimited_tag_string(self) -> None:
        """A delimited tag string should become a tag set."""
        comp = Competition.model_validate(self._payload(tags="tabular, regression; feature engineering"))

        assert comp.tags == frozenset({"tabular", "regression", "feature engineering"})

    def test_tag_sequence_is_unchanged(self) -> None:
        """A tag sequence should pass through untouched."""
        comp = Competition.model_validate(self._payload(tags=["tabular", "regression"]))

        assert comp.tags == frozenset({"tabular", "regression"})


class TestMetricDirectionFor:
    """Tests for the canonical metric direction mapping."""

    def test_every_metric_has_a_direction(self) -> None:
        """Each metric in the taxonomy should map to an explicit direction."""
        directions = {metric: metric_direction_for(metric) for metric in EvaluationMetric}

        assert set(directions.values()) == {"maximize", "minimize"}

    def test_error_metrics_minimize(self) -> None:
        """Error metrics should minimise."""
        assert metric_direction_for(EvaluationMetric.RMSE) == "minimize"
        assert metric_direction_for(EvaluationMetric.MAE) == "minimize"
        assert metric_direction_for(EvaluationMetric.RMSLE) == "minimize"
        assert metric_direction_for(EvaluationMetric.LOG_LOSS) == "minimize"

    def test_score_metrics_maximize(self) -> None:
        """Score and ranking metrics should maximise."""
        assert metric_direction_for(EvaluationMetric.ACCURACY) == "maximize"
        assert metric_direction_for(EvaluationMetric.AUC) == "maximize"
        assert metric_direction_for(EvaluationMetric.F1) == "maximize"
        assert metric_direction_for(EvaluationMetric.MAP) == "maximize"
        assert metric_direction_for(EvaluationMetric.NDCG) == "maximize"


class TestLeaderboardEntry:
    """Tests for the LeaderboardEntry model."""

    def test_creation(self) -> None:
        """Entry should be created with required fields."""
        entry = LeaderboardEntry(rank=1, team_name="winning_team", score=0.99999)

        assert entry.rank == 1
        assert entry.team_name == "winning_team"
        assert entry.score == 0.99999


class TestMissionCriteria:
    """Tests for the MissionCriteria model."""

    def test_default_values(self) -> None:
        """Criteria should have sensible defaults."""
        criteria = MissionCriteria()

        assert criteria.target_leaderboard_percentile == 0.10
        assert criteria.max_evolution_rounds == 100
        assert criteria.min_improvements_required == 0

    def test_custom_values(self) -> None:
        """Criteria should accept custom values."""
        criteria = MissionCriteria(
            target_leaderboard_percentile=0.05, max_evolution_rounds=200, min_improvements_required=10
        )

        assert criteria.target_leaderboard_percentile == 0.05
        assert criteria.max_evolution_rounds == 200
        assert criteria.min_improvements_required == 10


class TestMissionResult:
    """Tests for the MissionResult model."""

    def test_success_result(self) -> None:
        """Success result should have correct values."""
        result = MissionResult(
            success=True, mission_id="test_mission", competition_id="titanic", final_rank=10, final_score=0.85
        )

        assert result.success is True
        assert result.final_rank == 10
        assert result.final_score == 0.85

    def test_failure_result(self) -> None:
        """Failure result should include error message."""
        result = MissionResult(success=False, mission_id="test_mission", error_message="Discovery failed")

        assert result.success is False
        assert result.error_message == "Discovery failed"


class TestPlannedTask:
    """Tests for the PlannedTask model."""

    def test_creation(self) -> None:
        """Task should be created with required fields."""
        task = PlannedTask(
            id="task_1", name="Search competitions", description="Search Kaggle for competitions", agent="lobbyist"
        )

        assert task.id == "task_1"
        assert task.name == "Search competitions"
        assert task.description == "Search Kaggle for competitions"


class TestGenerationMetrics:
    """Tests for the GenerationMetrics model."""

    def test_creation(self) -> None:
        """Metrics should be created with required fields."""
        metrics = GenerationMetrics(
            generation=10, best_fitness=0.85, mean_fitness=0.75, worst_fitness=0.60, population_size=50
        )

        assert metrics.generation == 10
        assert metrics.best_fitness == 0.85
        assert metrics.mean_fitness == 0.75


class TestEvolutionState:
    """Tests for the EvolutionState model."""

    def test_default_creation(self) -> None:
        """State should be created with defaults."""
        state = EvolutionState()

        assert state.current_generation == 0
        assert state.convergence_detected is False
        assert state.best_solution is None
        assert state.improvement_count == 0
        assert state.min_improvements_required == 0
        assert state.failure_summary == {}


class TestMemoryEntry:
    """Tests for the MemoryEntry model."""

    def test_creation(self) -> None:
        """Entry should be created with required fields."""
        entry = MemoryEntry(key="test_key", category="test_category", value_preview="test preview")

        assert entry.key == "test_key"
        assert entry.category == "test_category"


class TestMemoryState:
    """Tests for the MemoryState model."""

    def test_default_creation(self) -> None:
        """State should be created with empty entries."""
        state = MemoryState()

        assert state.entries == []
        assert state.checkpoints == []


class TestErrorEvent:
    """Tests for the ErrorEvent model."""

    def test_creation(self) -> None:
        """Error should be created with required fields."""
        error = ErrorEvent(id="err_1", error_type="RateLimitError", category="recoverable", message="Rate limited")

        assert error.id == "err_1"
        assert error.error_type == "RateLimitError"
        assert error.message == "Rate limited"
