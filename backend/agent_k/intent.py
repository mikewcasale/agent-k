"""Classify mission intent without import-time model or network access.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from .models import CompetitionType, MissionCriteria as MissionCriteriaInput

__all__ = (
    "CompetitionType",
    "MissionCriteriaInput",
    "MissionIntentParser",
    "MissionIntentResult",
    "extract_latest_user_message",
    "parse_mission_intent",
)

_MISSION_ACTIONS: Final[tuple[str, ...]] = (
    "compete",
    "discover",
    "enter",
    "find",
    "join",
    "participate",
    "run",
    "search",
    "start",
)
_MISSION_OBJECTS: Final[tuple[str, ...]] = (
    "challenge",
    "competition",
    "kaggle",
    "mission",
)
_EXPLANATION_PREFIXES: Final[tuple[str, ...]] = (
    "explain",
    "how does",
    "tell me about",
    "what does",
    "what is",
)
_DOMAIN_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "audio": ("audio", "sound", "speech"),
    "computer_vision": ("computer vision", "image", "vision"),
    "finance": ("finance", "financial", "market", "trading"),
    "geospatial": ("geospatial", "gis", "satellite", "spatial"),
    "medical": ("clinical", "health", "healthcare", "medical"),
    "nlp": ("language", "natural language", "nlp", "text"),
    "tabular": ("csv", "structured data", "table", "tabular"),
    "time_series": ("forecast", "temporal", "time series", "timeseries"),
    "weather": ("climate", "weather"),
}
_PRIZE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(k|m|thousand|million)?\b",
    re.IGNORECASE,
)
_DAYS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:at\s+least\s+|minimum\s+)?(\d+)\s*(days?|weeks?)\b",
    re.IGNORECASE,
)
_PERCENTILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\btop\s+(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_ROUNDS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(\d+)\s*(?:evolution\s+)?(?:rounds?|generations?|iterations?)\b",
    re.IGNORECASE,
)


class MissionIntentResult(BaseModel):
    """Deterministic classification result for one user message."""

    model_config = ConfigDict(frozen=True)

    is_mission: bool
    criteria: MissionCriteriaInput | None = None


@dataclass(frozen=True, slots=True)
class MissionIntentParser:
    """Extract mission intent and criteria using deterministic language rules."""

    mission_actions: tuple[str, ...] = _MISSION_ACTIONS
    mission_objects: tuple[str, ...] = _MISSION_OBJECTS

    def parse(self, message: str) -> MissionIntentResult:
        """Classify one message and extract supported mission criteria.

        Args:
            message: User-authored text from the latest chat message.

        Returns:
            A classification and validated criteria for mission requests.
        """
        normalized = " ".join(message.lower().split())
        if not normalized or normalized.startswith(_EXPLANATION_PREFIXES):
            return MissionIntentResult(is_mission=False)

        has_action = any(_contains_term(normalized, term) for term in self.mission_actions)
        has_object = any(_contains_term(normalized, term) for term in self.mission_objects)
        if not (has_action and has_object):
            return MissionIntentResult(is_mission=False)

        return MissionIntentResult(
            is_mission=True,
            criteria=_extract_criteria(normalized),
        )


def parse_mission_intent(
    messages: list[dict[str, Any]],
    *,
    parser: MissionIntentParser | None = None,
) -> MissionIntentResult:
    """Parse the latest user text from a frontend chat message list.

    Args:
        messages: Vercel AI-style messages with text parts.
        parser: Optional deterministic parser configuration.

    Returns:
        A non-mission result when no usable user text is present.
    """
    message = extract_latest_user_message(messages)
    if message is None:
        return MissionIntentResult(is_mission=False)
    return (parser or MissionIntentParser()).parse(message)


def extract_latest_user_message(messages: list[dict[str, Any]]) -> str | None:
    """Return text from the latest user message without accepting other roles."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            return None
        text_parts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        text = " ".join(text_parts).strip()
        return text or None
    return None


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _extract_criteria(text: str) -> MissionCriteriaInput:
    values: dict[str, Any] = {}

    competition_types: set[CompetitionType] = set()
    candidates: tuple[CompetitionType, ...] = (
        CompetitionType.FEATURED,
        CompetitionType.RESEARCH,
        CompetitionType.PLAYGROUND,
        CompetitionType.GETTING_STARTED,
        CompetitionType.COMMUNITY,
    )
    for candidate in candidates:
        phrase = candidate.value.replace("_", " ")
        if _contains_term(text, phrase):
            competition_types.add(candidate)
    if competition_types:
        values["target_competition_types"] = competition_types

    if prize_match := _PRIZE_PATTERN.search(text):
        amount = float(prize_match.group(1).replace(",", ""))
        suffix = (prize_match.group(2) or "").lower()
        if suffix in {"k", "thousand"}:
            amount *= 1_000
        elif suffix in {"m", "million"}:
            amount *= 1_000_000
        values["min_prize_pool"] = int(amount)

    if days_match := _DAYS_PATTERN.search(text):
        days = int(days_match.group(1))
        if days_match.group(2).lower().startswith("week"):
            days *= 7
        values["min_days_remaining"] = days

    domains = {
        domain
        for domain, keywords in _DOMAIN_KEYWORDS.items()
        if any(_contains_term(text, keyword) for keyword in keywords)
    }
    if domains:
        values["target_domains"] = domains

    if percentile_match := _PERCENTILE_PATTERN.search(text):
        values["target_leaderboard_percentile"] = min(
            float(percentile_match.group(1)) / 100,
            1.0,
        )

    if rounds_match := _ROUNDS_PATTERN.search(text):
        values["max_evolution_rounds"] = int(rounds_match.group(1))

    return MissionCriteriaInput(**values)
