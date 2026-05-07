"""Domain matching utilities for competition discovery.

@notice: |
    Domain matching utilities for competition discovery.

@dev: |
    Centralizes the keyword expansion and token-aware matching used by both
    the Lobbyist scoring tool and the AG-UI competition filter so the two
    code paths agree on what "this competition matches that domain" means.

@graph:
    id: agent_k.core.discovery
    provides:
        - agent_k.core.discovery:DOMAIN_KEYWORDS
        - agent_k.core.discovery:match_competition_domains
        - agent_k.core.discovery:normalize_domain_key
    pattern: domain-matching

@agent-guidance:
    do:
        - "Use agent_k.core.discovery as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-05-07
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re
from typing import Annotated, Final

from agent_k.core.models import Competition
from agent_k.core.sage import Doc

__all__ = ("DOMAIN_KEYWORDS", "match_competition_domains", "normalize_domain_key")

DOMAIN_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "finance": ("finance", "financial", "trading", "stock", "market"),
    "medical": ("medical", "health", "healthcare", "clinical", "diagnosis"),
    "weather": ("weather", "climate", "forecast"),
    "computer_vision": ("computer vision", "vision", "image", "cv"),
    "nlp": ("nlp", "text", "language", "transformer"),
    "tabular": ("tabular", "structured", "csv", "table"),
    "time_series": ("time series", "timeseries", "temporal", "forecast"),
    "audio": ("audio", "speech", "sound", "acoustic"),
    "geospatial": ("geospatial", "geo", "spatial", "gis", "satellite"),
}
"""Canonical domain → keyword expansions used for tag/title/description matching."""

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


def normalize_domain_key(domain: Annotated[str, Doc("Raw domain string from agent or UI input.")]) -> str:
    """Normalize a domain label for ``DOMAIN_KEYWORDS`` lookup.

    @dev: |
        Lowercases, strips, and collapses whitespace and hyphens to underscores
        so variants like ``"Computer Vision"`` and ``"computer-vision"`` both
        resolve to ``"computer_vision"``.
    """
    return domain.strip().lower().replace(" ", "_").replace("-", "_")


def match_competition_domains(
    competition: Annotated[Competition, Doc("Competition to evaluate.")],
    domains: Annotated[list[str], Doc("Target domain labels supplied by the agent or user.")],
) -> bool:
    """Return ``True`` when ``competition`` matches any requested domain.

    @notice: |
        Token-aware match against tags, title, and description.

    @dev: |
        Matching strategy per keyword:
        - Tag set: case-insensitive equality (tags are curated phrases).
        - Multi-word keywords (e.g. ``"computer vision"``): substring match
          against the lowercased title+description haystack so phrases survive
          intact.
        - Single-word keywords: token-level membership against the haystack.
          This avoids the historical substring false positives where
          ``"ai"`` matched ``"audio"`` or ``"ml"`` matched ``"html"``.

        An empty ``domains`` list always matches (no filter applied).
    """
    if not domains:
        return True

    tags = {tag.lower() for tag in competition.tags}
    haystack = f"{competition.title or ''} {competition.description or ''}".lower()
    haystack_tokens = set(_TOKEN_PATTERN.findall(haystack))

    for domain in domains:
        key = normalize_domain_key(domain)
        keywords = DOMAIN_KEYWORDS.get(key, (key,))
        for raw_keyword in keywords:
            keyword = raw_keyword.lower().strip()
            if not keyword:
                continue
            if keyword in tags:
                return True
            if " " in keyword:
                if keyword in haystack:
                    return True
            elif keyword in haystack_tokens:
                return True
    return False
