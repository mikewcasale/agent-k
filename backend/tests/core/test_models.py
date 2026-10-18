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

    def test_defaults_are_enterable(self) -> None:
        """A competition with no eligibility flags should be enterable."""
        comp = Competition(
            id="titanic",
            title="Titanic",
            competition_type=CompetitionType.GETTING_STARTED,
            metric=EvaluationMetric.ACCURACY,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
        )

        assert comp.accepts_file_submissions is True
        assert comp.accepts_new_entrants is True
        assert comp.is_enterable is True
        assert comp.entry_blockers == ()

    def test_kernels_only_blocks_file_submissions(self) -> None:
        """A code competition should not accept file submissions."""
        comp = Competition(
            id="code-comp",
            title="Code competition",
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.AUC,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
            kernels_submissions_only=True,
        )

        assert comp.accepts_file_submissions is False
        assert comp.is_enterable is False
        assert comp.entry_blockers == ("kernels_submissions_only",)

    def test_disabled_submissions_block_entry(self) -> None:
        """Disabled submissions should block entry even before the deadline."""
        comp = Competition(
            id="paused-comp",
            title="Paused competition",
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.RMSE,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
            submissions_disabled=True,
        )

        assert comp.accepts_file_submissions is False
        assert comp.entry_blockers == ("submissions_disabled",)

    def test_passed_new_entrant_deadline_blocks_entry(self) -> None:
        """A closed new-entrant window should block entry for a non-participant."""
        comp = Competition(
            id="closed-entry",
            title="Closed entry",
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.AUC,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
            new_entrant_deadline=datetime(2020, 1, 1, tzinfo=UTC),
        )

        assert comp.accepts_new_entrants is False
        assert comp.entry_blockers == ("new_entrant_deadline_passed",)

    def test_entered_user_ignores_new_entrant_deadline(self) -> None:
        """An already-entered competition stays enterable after the entry window."""
        comp = Competition(
            id="already-entered",
            title="Already entered",
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.AUC,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
            new_entrant_deadline=datetime(2020, 1, 1, tzinfo=UTC),
            user_has_entered=True,
        )

        assert comp.accepts_new_entrants is True
        assert comp.is_enterable is True

    def test_expired_deadline_blocks_entry(self) -> None:
        """A passed deadline should be reported as an entry blocker."""
        comp = Competition(
            id="expired",
            title="Expired",
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.AUC,
            deadline=datetime(2020, 1, 1, tzinfo=UTC),
        )

        assert comp.is_active is False
        assert comp.entry_blockers == ("deadline_passed",)

    def test_naive_new_entrant_deadline_rejected(self) -> None:
        """A timezone-naive new-entrant deadline should be rejected."""
        with pytest.raises(ValidationError):
            Competition(
                id="naive-entry",
                title="Naive entry",
                competition_type=CompetitionType.FEATURED,
                metric=EvaluationMetric.AUC,
                deadline=datetime(2030, 1, 1, tzinfo=UTC),
                new_entrant_deadline=datetime(2029, 1, 1),
            )

    def test_multiple_blockers_reported(self) -> None:
        """Every applicable blocker should be reported."""
        comp = Competition(
            id="hopeless",
            title="Hopeless",
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.AUC,
            deadline=datetime(2020, 1, 1, tzinfo=UTC),
            submissions_disabled=True,
            kernels_submissions_only=True,
        )

        assert comp.entry_blockers == ("deadline_passed", "submissions_disabled", "kernels_submissions_only")


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
