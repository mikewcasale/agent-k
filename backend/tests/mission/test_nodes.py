"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_k.core.models import EvaluationMetric, LeaderboardEntry, Submission
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _evaluate_metric,
    _fetch_leaderboard_safely,
    _find_rank_for_score,
    _poll_submission_score,
)

__all__ = ()

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real sleeps inside the SubmissionNode helpers to keep tests fast."""

    async def _noop(_: float) -> None:
        return None

    monkeypatch.setattr("agent_k.mission.nodes.asyncio.sleep", _noop)


def _submission(**overrides: Any) -> Submission:
    defaults: dict[str, Any] = {
        "id": "sub-123",
        "competition_id": "comp-1",
        "file_name": "submission.csv",
        "submitted_at": datetime(2026, 1, 1, tzinfo=UTC),
        "status": "pending",
    }
    defaults.update(overrides)
    return Submission(**defaults)


def _entry(rank: int, score: float, team_name: str | None = None) -> LeaderboardEntry:
    return LeaderboardEntry(rank=rank, team_name=team_name or f"team-{rank}", score=score)


class TestDiscoveryNode:
    """Tests for the DiscoveryNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = DiscoveryNode()
        assert node is not None


class TestResearchNode:
    """Tests for the ResearchNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = ResearchNode()
        assert node is not None


class TestPrototypeNode:
    """Tests for the PrototypeNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = PrototypeNode()
        assert node is not None


class TestEvolutionNode:
    """Tests for the EvolutionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = EvolutionNode()
        assert node is not None


class TestSubmissionNode:
    """Tests for the SubmissionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = SubmissionNode()
        assert node is not None


class TestEvaluateMetric:
    """Tests for metric evaluation helpers."""

    def test_rmsle_ignores_negative_values(self) -> None:
        """RMSLE should ignore negative targets in the denominator."""
        score = _evaluate_metric(EvaluationMetric.RMSLE, [1.0, -1.0], prediction=0.0)
        assert score == pytest.approx(math.log1p(1.0))


class TestPollSubmissionScore:
    """Tests for ``_poll_submission_score``."""

    async def test_returns_score_on_first_success(self) -> None:
        adapter = AsyncMock()
        adapter.get_submission_status = AsyncMock(return_value=_submission(status="complete", public_score=0.87))

        score = await _poll_submission_score(
            adapter, competition_id="comp-1", submission_id="sub-1", attempts=5, interval_seconds=0.0
        )

        assert score == pytest.approx(0.87)
        assert adapter.get_submission_status.await_count == 1

    async def test_polls_until_score_appears(self) -> None:
        adapter = AsyncMock()
        adapter.get_submission_status = AsyncMock(
            side_effect=[
                _submission(status="pending", public_score=None),
                _submission(status="pending", public_score=None),
                _submission(status="complete", public_score=0.42),
            ]
        )

        score = await _poll_submission_score(
            adapter, competition_id="comp-1", submission_id="sub-1", attempts=5, interval_seconds=0.0
        )

        assert score == pytest.approx(0.42)
        assert adapter.get_submission_status.await_count == 3

    async def test_transient_error_does_not_abort_polling(self) -> None:
        adapter = AsyncMock()
        adapter.get_submission_status = AsyncMock(
            side_effect=[RuntimeError("kaggle 503"), _submission(status="complete", public_score=0.61)]
        )

        score = await _poll_submission_score(
            adapter, competition_id="comp-1", submission_id="sub-1", attempts=5, interval_seconds=0.0
        )

        assert score == pytest.approx(0.61)
        assert adapter.get_submission_status.await_count == 2

    async def test_returns_none_when_all_polls_error(self) -> None:
        adapter = AsyncMock()
        adapter.get_submission_status = AsyncMock(side_effect=RuntimeError("kaggle down"))

        score = await _poll_submission_score(
            adapter, competition_id="comp-1", submission_id="sub-1", attempts=3, interval_seconds=0.0
        )

        assert score is None
        assert adapter.get_submission_status.await_count == 3

    async def test_returns_none_when_score_never_arrives(self) -> None:
        adapter = AsyncMock()
        adapter.get_submission_status = AsyncMock(return_value=_submission(status="pending", public_score=None))

        score = await _poll_submission_score(
            adapter, competition_id="comp-1", submission_id="sub-1", attempts=4, interval_seconds=0.0
        )

        assert score is None
        assert adapter.get_submission_status.await_count == 4


class TestFetchLeaderboardSafely:
    """Tests for ``_fetch_leaderboard_safely``."""

    async def test_returns_entries_on_success(self) -> None:
        entries = [_entry(1, 0.9), _entry(2, 0.8)]
        adapter = AsyncMock()
        adapter.get_leaderboard = AsyncMock(return_value=entries)

        result = await _fetch_leaderboard_safely(adapter, competition_id="comp-1", limit=100)

        assert result == entries
        adapter.get_leaderboard.assert_awaited_once_with("comp-1", limit=100)

    async def test_returns_empty_on_error(self) -> None:
        adapter = AsyncMock()
        adapter.get_leaderboard = AsyncMock(side_effect=RuntimeError("kaggle 500"))

        result = await _fetch_leaderboard_safely(adapter, competition_id="comp-1", limit=100)

        assert result == []


class TestFindRankForScore:
    """Tests for ``_find_rank_for_score``."""

    def test_exact_match_returns_rank(self) -> None:
        entries = [_entry(1, 0.9), _entry(2, 0.8), _entry(3, 0.7)]

        assert _find_rank_for_score(entries, 0.8) == 2

    def test_matches_within_float_tolerance(self) -> None:
        # 0.1 + 0.2 != 0.3 exactly in IEEE 754; the tolerant match must still find it.
        entries = [_entry(1, 0.5), _entry(2, 0.1 + 0.2), _entry(3, 0.4)]

        assert _find_rank_for_score(entries, 0.3) == 2

    def test_returns_none_when_no_match(self) -> None:
        entries = [_entry(1, 0.9), _entry(2, 0.8)]

        assert _find_rank_for_score(entries, 0.5) is None

    def test_returns_none_for_empty_leaderboard(self) -> None:
        assert _find_rank_for_score([], 0.5) is None

    def test_returns_first_matching_rank_on_ties(self) -> None:
        entries = [_entry(1, 0.9), _entry(2, 0.8), _entry(3, 0.8)]

        assert _find_rank_for_score(entries, 0.8) == 2
