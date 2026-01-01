"""Tests for the graph state models.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agent_k.core.models import MissionCriteria
from agent_k.mission.state import GraphContext, MissionState

__all__ = ()

pytestmark = pytest.mark.anyio


class TestMissionState:
    """Tests for the MissionState model."""

    def test_creation_with_mission_id(self) -> None:
        """State should be created with mission ID."""
        mission_id = str(uuid4())
        state = MissionState(mission_id=mission_id)

        assert state.mission_id == mission_id
        assert state.competition_id is None
        assert state.current_phase == "discovery"

    def test_creation_with_competition_id(self) -> None:
        """State should accept competition ID."""
        state = MissionState(
            mission_id=str(uuid4()),
            competition_id="titanic",
        )

        assert state.competition_id == "titanic"

    def test_default_criteria(self) -> None:
        """State should have default criteria."""
        state = MissionState(mission_id=str(uuid4()))

        assert state.criteria is not None
        assert isinstance(state.criteria, MissionCriteria)

    def test_custom_criteria(self) -> None:
        """State should accept custom criteria."""
        criteria = MissionCriteria(
            target_leaderboard_percentile=0.05,
            max_evolution_rounds=50,
        )
        state = MissionState(
            mission_id=str(uuid4()),
            criteria=criteria,
        )

        assert state.criteria.target_leaderboard_percentile == 0.05
        assert state.criteria.max_evolution_rounds == 50

    def test_phases_completed_default(self) -> None:
        """Phases completed should default to empty list."""
        state = MissionState(mission_id=str(uuid4()))

        assert state.phases_completed == []

    def test_discovered_competitions_default(self) -> None:
        """Discovered competitions should default to empty list."""
        state = MissionState(mission_id=str(uuid4()))

        assert state.discovered_competitions == []

    def test_phase_results_none_by_default(self) -> None:
        """Phase results should be None by default."""
        state = MissionState(mission_id=str(uuid4()))

        assert state.selected_competition is None
        assert state.research_findings is None
        assert state.prototype_code is None
        assert state.evolution_state is None
        assert state.final_submission_id is None
        assert state.final_score is None
        assert state.final_rank is None

    def test_started_at_auto_set(self) -> None:
        """Started at should be auto-set to current time."""
        before = datetime.now(UTC)
        state = MissionState(mission_id=str(uuid4()))
        after = datetime.now(UTC)

        assert before <= state.started_at <= after

    def test_overall_progress_default(self) -> None:
        """Overall progress should default to 0."""
        state = MissionState(mission_id=str(uuid4()))

        assert state.overall_progress == 0.0


class TestGraphContext:
    """Tests for the GraphContext model."""

    def test_empty_context(self) -> None:
        """Context should be creatable with no arguments."""
        context = GraphContext()

        assert context.event_emitter is None
        assert context.http_client is None
        assert context.platform_adapter is None

    def test_context_with_emitter(self) -> None:
        """Context should accept event emitter."""
        emitter = MagicMock()

        context = GraphContext(event_emitter=emitter)

        assert context.event_emitter is emitter
