"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _evaluate_metric,
    _format_entry_blockers,
    _select_enterable_competition,
)

__all__ = ()

pytestmark = pytest.mark.anyio


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


def _competition(competition_id: str, **overrides: Any) -> Competition:
    """Build a competition with permissive defaults."""
    fields: dict[str, Any] = {
        "id": competition_id,
        "title": "Example",
        "competition_type": CompetitionType.FEATURED,
        "metric": EvaluationMetric.AUC,
        "deadline": datetime(2030, 1, 1, tzinfo=UTC),
    }
    fields.update(overrides)
    return Competition(**fields)


class TestSelectEnterableCompetition:
    """Tests for competition selection in the discovery phase."""

    def test_skips_unenterable_candidates(self) -> None:
        """Selection should skip competitions that cannot be submitted to."""
        code_only = _competition("code-comp", kernels_submissions_only=True)
        open_comp = _competition("open-comp")

        selected, blockers = _select_enterable_competition([code_only, open_comp], {})

        assert selected is not None
        assert selected.id == "open-comp"
        assert blockers == {"code-comp": ["kernels_submissions_only"]}

    def test_cached_metadata_overrides_model_output(self) -> None:
        """Platform metadata should win over model-emitted eligibility flags."""
        model_output = _competition("code-comp")
        cached = _competition("code-comp", kernels_submissions_only=True)

        selected, blockers = _select_enterable_competition([model_output], {"code-comp": cached})

        assert selected is None
        assert blockers == {"code-comp": ["kernels_submissions_only"]}

    def test_cached_metadata_is_returned(self) -> None:
        """The authoritative cached competition should be returned when present."""
        model_output = _competition("open-comp", max_daily_submissions=5)
        cached = _competition("open-comp", max_daily_submissions=2)

        selected, blockers = _select_enterable_competition([model_output], {"open-comp": cached})

        assert selected is cached
        assert blockers == {}

    def test_non_competition_cache_entry_ignored(self) -> None:
        """A cache entry that is not a Competition should be ignored."""
        model_output = _competition("open-comp")

        selected, _blockers = _select_enterable_competition([model_output], {"open-comp": {"id": "open-comp"}})

        assert selected is model_output

    def test_no_candidates_returns_none(self) -> None:
        """An empty candidate list should report no blockers."""
        selected, blockers = _select_enterable_competition([], {})

        assert selected is None
        assert blockers == {}


class TestFormatEntryBlockers:
    """Tests for rendering entry blockers into a diagnostic string."""

    def test_empty_blockers(self) -> None:
        """No blockers should read as no competitions returned."""
        assert _format_entry_blockers({}) == "no competitions returned"

    def test_blockers_rendered_per_competition(self) -> None:
        """Each competition should be rendered with its reasons."""
        rendered = _format_entry_blockers({"a": ["deadline_passed"], "b": ["kernels_submissions_only"]})

        assert rendered == "a: deadline_passed; b: kernels_submissions_only"
