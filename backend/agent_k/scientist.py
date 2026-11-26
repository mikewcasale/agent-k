"""Evidence-bound research for an already discovered dataset.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from .hints import DatasetProfile
from .interfaces import PlatformAdapter
from .models import Competition, LeaderboardEntry

__all__ = (
    "ResearchFinding",
    "ResearchReport",
    "ScientistDeps",
    "scientist_agent",
    "scientist_toolset",
)

SCIENTIST_SYSTEM_PROMPT = """Research the supplied competition and training-only profile.
Use leaderboard and paper tools for evidence. Cite only observed URLs. Recommend
generic techniques appropriate to the six supported classification/regression
problem types. Prefer LightGBM and test a real custom gradient/Hessian objective.
For tabular data, consider fixed pretrained NVIDIA Kumo beside LightGBM, subject
to external-data rules and resource limits. Do not claim measured improvement
before independent evaluation. Never use acceptance labels or infer a public rank
from local fitness. External data is disallowed unless rules explicitly permit it.
"""


class ResearchFinding(BaseModel):
    """A recommendation with an observed source and a measurable validation step."""

    model_config = ConfigDict(frozen=True)
    recommendation: str
    source: HttpUrl
    validation: str


class ResearchReport(BaseModel):
    """Research output, separate from measured evaluation results."""

    model_config = ConfigDict(frozen=True)
    summary: str
    findings: tuple[ResearchFinding, ...] = ()
    recommendations: tuple[str, ...] = Field(min_length=1)
    external_data_allowed: bool = False
    external_data_evidence: str | None = None


@dataclass
class ScientistDeps:
    """Research services and training-only evidence for one mission."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    competition: Competition
    profile: DatasetProfile
    observed_urls: set[str] = field(default_factory=set)


class _Paper(BaseModel):
    title: str
    url: str
    summary: str


scientist_toolset: FunctionToolset[ScientistDeps] = FunctionToolset(id="scientist")


@scientist_toolset.tool
async def leaderboard(ctx: RunContext[ScientistDeps]) -> list[LeaderboardEntry]:
    """Inspect actual public leaderboard values without estimating missing values."""
    return await ctx.deps.platform_adapter.get_leaderboard(ctx.deps.competition.id, limit=100)


@scientist_toolset.tool
async def search_papers(ctx: RunContext[ScientistDeps], query: str) -> list[_Paper]:
    """Search arXiv for a bounded set of relevant primary research sources."""
    response = await ctx.deps.http_client.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": "5"},
        timeout=30,
    )
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("Paper response exceeds byte limit")
    root = ET.fromstring(response.content)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[_Paper] = []
    for entry in root.findall("atom:entry", namespace):
        url = entry.findtext("atom:id", namespaces=namespace)
        title = entry.findtext("atom:title", namespaces=namespace)
        summary = entry.findtext("atom:summary", namespaces=namespace)
        if not url or not title or not summary:
            continue
        papers.append(_Paper(title=title.strip(), url=url, summary=summary.strip()))
        ctx.deps.observed_urls.add(url)
    return papers


scientist_agent: Agent[ScientistDeps, ResearchReport] = Agent(
    model=None,
    deps_type=ScientistDeps,
    output_type=ResearchReport,
    instructions=SCIENTIST_SYSTEM_PROMPT,
    toolsets=[scientist_toolset],
    name="scientist",
    retries=2,
)


@scientist_agent.output_validator
def _validate_evidence(ctx: RunContext[ScientistDeps], report: ResearchReport) -> ResearchReport:
    if any(str(finding.source) not in ctx.deps.observed_urls for finding in report.findings):
        raise ModelRetry("Every research citation must come from the paper search tool")
    if report.external_data_allowed:
        raise ModelRetry("No external-data rules have been verified by this research toolset")
    return report
