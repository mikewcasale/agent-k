"""Competition discovery with runtime dependency injection.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from .interfaces import PlatformAdapter
from .models import Competition, MissionCriteria

__all__ = ("DiscoveryResult", "LobbyistDeps", "discover", "lobbyist_agent", "lobbyist_toolset")

LOBBYIST_SYSTEM_PROMPT = """Discover real, active competitions using the search tool.
Respect the supplied mission criteria. Select only competition IDs returned by the
tool. Explain the choice using observed metadata. Subject tags do not establish
support for an ML task; dataset discovery later verifies the problem type.
Never invent competitions, scores, accepted rules, or download permissions.
"""


@dataclass
class LobbyistDeps:
    """Services and evidence scoped to one discovery run."""

    platform_adapter: PlatformAdapter
    criteria: MissionCriteria
    observed: dict[str, Competition] = field(default_factory=dict)


class DiscoveryResult(BaseModel):
    """Selection constrained to an observed competition identifier."""

    model_config = ConfigDict(frozen=True)
    competition_id: str
    rationale: str = Field(min_length=1)


lobbyist_toolset: FunctionToolset[LobbyistDeps] = FunctionToolset(id="lobbyist")


@lobbyist_toolset.tool
async def discover(ctx: RunContext[LobbyistDeps]) -> list[Competition]:
    """Find real competitions satisfying the user's hard constraints."""
    criteria = ctx.deps.criteria
    found: list[Competition] = []
    async for competition in ctx.deps.platform_adapter.search_competitions(
        categories=[category.value for category in criteria.target_competition_types],
        min_prize=criteria.min_prize_pool,
    ):
        if competition.days_remaining < criteria.min_days_remaining:
            continue
        if criteria.max_team_size and competition.max_team_size > criteria.max_team_size:
            continue
        if criteria.exclude_domains & competition.tags:
            continue
        if criteria.target_domains and not criteria.target_domains & competition.tags:
            continue
        found.append(competition)
        ctx.deps.observed[competition.id] = competition
        if len(found) >= 50:
            break
    return found


lobbyist_agent: Agent[LobbyistDeps, DiscoveryResult] = Agent(
    model=None,
    deps_type=LobbyistDeps,
    output_type=DiscoveryResult,
    instructions=LOBBYIST_SYSTEM_PROMPT,
    toolsets=[lobbyist_toolset],
    name="lobbyist",
    retries=2,
)


@lobbyist_agent.output_validator
def _validate_selection(ctx: RunContext[LobbyistDeps], output: DiscoveryResult) -> DiscoveryResult:
    if output.competition_id not in ctx.deps.observed:
        raise ModelRetry("Select an ID returned by the discovery tool")
    return output
