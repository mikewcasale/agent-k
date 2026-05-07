"""Tests for token-aware competition domain matching.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime

import pytest

from agent_k.core.discovery import DOMAIN_KEYWORDS, match_competition_domains, normalize_domain_key
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric

__all__ = ()


def _competition(
    *, tags: tuple[str, ...] = (), title: str = "Sample Competition", description: str | None = None
) -> Competition:
    return Competition(
        id="sample",
        title=title,
        description=description,
        competition_type=CompetitionType.FEATURED,
        metric=EvaluationMetric.ACCURACY,
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
        tags=frozenset(tags),
    )


class TestNormalizeDomainKey:
    """Tests for ``normalize_domain_key``."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Computer Vision", "computer_vision"),
            ("computer-vision", "computer_vision"),
            ("  Time  Series  ", "time__series"),
            ("NLP", "nlp"),
            ("Tabular", "tabular"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        """Whitespace, hyphens, and casing collapse to a stable lookup key."""
        assert normalize_domain_key(raw) == expected


class TestMatchCompetitionDomains:
    """Tests for token-aware domain matching against tags and text."""

    def test_empty_domains_always_matches(self) -> None:
        """No filter requested → match everything."""
        comp = _competition(tags=("tabular",))
        assert match_competition_domains(comp, []) is True

    def test_tag_membership_matches(self) -> None:
        """A competition tagged ``tabular`` matches the ``tabular`` domain."""
        comp = _competition(tags=("tabular",))
        assert match_competition_domains(comp, ["tabular"]) is True

    def test_tag_membership_is_case_insensitive(self) -> None:
        """Tag matching ignores case differences from the source data."""
        comp = _competition(tags=("Tabular",))
        assert match_competition_domains(comp, ["TABULAR"]) is True

    def test_keyword_expansion_via_tags(self) -> None:
        """``computer_vision`` expands to keywords that match a CV-tagged comp."""
        comp = _competition(tags=("computer vision",))
        assert match_competition_domains(comp, ["computer_vision"]) is True

    def test_keyword_expansion_via_title_phrase(self) -> None:
        """Multi-word keywords match via substring on title/description."""
        comp = _competition(title="Time Series Forecasting Challenge", tags=())
        assert match_competition_domains(comp, ["time_series"]) is True

    def test_single_word_unknown_domain_uses_token_match(self) -> None:
        """Unknown single-word domains must match as whole tokens, not substrings.

        Regression: previously substring matching let ``"ai"`` match titles
        containing ``"audio"`` or ``"main"``, fabricating domain hits.
        """
        comp = _competition(title="Audio Classification Challenge", tags=("audio",))
        assert match_competition_domains(comp, ["ai"]) is False
        # And ``"ml"`` must not match titles containing ``"html"``.
        comp_html = _competition(title="HTML Layout Prediction", tags=())
        assert match_competition_domains(comp_html, ["ml"]) is False

    def test_unknown_single_word_matches_as_token(self) -> None:
        """An unknown domain still matches when it appears as a whole token."""
        comp = _competition(title="Quantum State Reconstruction", tags=())
        assert match_competition_domains(comp, ["quantum"]) is True

    def test_no_match_returns_false(self) -> None:
        """Unrelated competitions with no tag/text overlap return False."""
        comp = _competition(title="Image Segmentation", tags=("computer vision",))
        assert match_competition_domains(comp, ["finance"]) is False

    def test_description_haystack_used(self) -> None:
        """Description text contributes to the haystack for token matching."""
        comp = _competition(title="Generic Title", description="Predict daily temperature climate trends.", tags=())
        assert match_competition_domains(comp, ["weather"]) is True

    def test_missing_description_does_not_crash(self) -> None:
        """``None`` descriptions are tolerated without raising."""
        comp = _competition(title="Tabular Sales Prediction", description=None, tags=())
        assert match_competition_domains(comp, ["tabular"]) is True

    def test_multiple_domains_any_match(self) -> None:
        """Any matching domain returns True."""
        comp = _competition(tags=("nlp",))
        assert match_competition_domains(comp, ["finance", "nlp", "audio"]) is True

    def test_empty_string_keyword_does_not_match_everything(self) -> None:
        """A blank domain entry must not collapse to a wildcard match.

        ``""`` would have an empty keyword expansion; matching it against any
        haystack would otherwise short-circuit to True via ``"" in haystack``.
        """
        comp = _competition(title="Some Title", tags=("audio",))
        assert match_competition_domains(comp, [""]) is False

    def test_domain_keywords_contains_expected_keys(self) -> None:
        """Public ``DOMAIN_KEYWORDS`` exposes the canonical domain set."""
        assert {"finance", "medical", "weather", "computer_vision", "nlp"}.issubset(DOMAIN_KEYWORDS.keys())
