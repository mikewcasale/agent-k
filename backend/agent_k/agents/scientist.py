"""Scientist agent - research and analysis for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import csv
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

# Third-party (alphabetical)
import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.agents import register_agent
from agent_k.agents.base import universal_tool_preparation
from agent_k.agents.prompts import SCIENTIST_SYSTEM_PROMPT
from agent_k.core.constants import DEFAULT_MODEL
from agent_k.infra.providers import get_model
from agent_k.toolsets import (
    AgentKMemoryTool,
    create_memory_backend,
    create_production_toolset,
    kaggle_toolset,
    prepare_memory_tool,
    prepare_web_search,
    register_memory_tool,
)

if TYPE_CHECKING:
    import httpx

    from agent_k.core.models import Competition, LeaderboardEntry
    from agent_k.core.protocols import PlatformAdapter

__all__ = (
    "LeaderboardAnalysis",
    "ResearchFinding",
    "ResearchReport",
    "ScientistAgent",
    "ScientistDeps",
    "ScientistSettings",
    "SCIENTIST_SYSTEM_PROMPT",
    "SCHEMA_VERSION",
    "scientist_agent",
)

SCHEMA_VERSION: Final[str] = "1.0.0"
_KAGGLE_KERNELS_ENDPOINT: Final[str] = "https://www.kaggle.com/api/v1/kernels/list"
_DEFAULT_NOTEBOOK_TECHNIQUES: Final[dict[str, str]] = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "random forest": "random_forest",
    "gradient boost": "gradient_boosting",
    "feature": "feature_engineering",
    "cross validation": "cross_validation",
    "cv": "cross_validation",
    "stack": "stacking",
}
_MISSING_VALUE_TOKENS: Final[frozenset[str]] = frozenset({"", "na", "nan", "null", "none"})


class ScientistSettings(BaseSettings):
    """Configuration for the Scientist agent."""

    model_config = SettingsConfigDict(
        env_prefix="SCIENTIST_",
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model identifier for research tasks",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for research prompts",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens for responses",
    )

    tool_retries: int = Field(
        default=2,
        ge=0,
        description="Tool retry attempts",
    )
    output_retries: int = Field(
        default=1,
        ge=0,
        description="Output validation retry attempts",
    )
    max_paper_results: int = Field(
        default=10,
        ge=1,
        description="Maximum papers to retrieve",
    )
    max_notebook_results: int = Field(
        default=10,
        ge=1,
        description="Maximum notebooks to retrieve",
    )

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings for the configured model."""
        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


@dataclass
class ScientistDeps:
    """Dependencies for the Scientist agent."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    competition: Competition
    leaderboard: list[LeaderboardEntry] = field(default_factory=list)
    research_cache: dict[str, Any] = field(default_factory=dict)

    async def refresh_leaderboard(self) -> None:
        """Refresh leaderboard from the platform."""
        self.leaderboard = await self.platform_adapter.get_leaderboard(
            self.competition.id,
            limit=100,
        )


class ResearchFinding(BaseModel):
    """Individual research finding."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    category: str = Field(description="Category of finding")
    title: str = Field(description="Brief title")
    summary: str = Field(description="Detailed summary")
    relevance_score: float = Field(ge=0, le=1, description="Relevance to competition")
    sources: list[str] = Field(default_factory=list, description="Source URLs")


class LeaderboardAnalysis(BaseModel):
    """Analysis of competition leaderboard."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    top_score: float = Field(description="Best leaderboard score")
    median_score: float = Field(description="Median leaderboard score")
    score_distribution: str = Field(description="Description of score distribution")
    common_approaches: list[str] = Field(description="Inferred common approaches")
    improvement_opportunities: list[str] = Field(description="Potential improvement areas")


class ResearchReport(BaseModel):
    """Complete research report for a competition."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    competition_id: str = Field(description="Competition identifier")
    domain_findings: list[ResearchFinding] = Field(
        default_factory=list,
        description="Domain-specific research findings",
    )
    technique_findings: list[ResearchFinding] = Field(
        default_factory=list,
        description="Technique-focused research findings",
    )
    leaderboard_analysis: LeaderboardAnalysis | None = Field(
        default=None,
        description="Leaderboard analysis summary",
    )
    recommended_approaches: list[str] = Field(
        default_factory=list,
        description="Recommended modeling approaches",
    )
    estimated_baseline_score: float | None = Field(
        default=None,
        description="Estimated baseline score",
    )
    key_challenges: list[str] = Field(
        default_factory=list,
        description="Primary competition challenges",
    )


class ScientistAgent:
    """Scientist agent encapsulating research and analysis functionality."""

    def __init__(self, settings: ScientistSettings | None = None) -> None:
        """Initialize the Scientist agent.

        Args:
            settings: Configuration for the agent. Uses defaults if not provided.
        """
        self._settings = settings or ScientistSettings()
        self._toolset: FunctionToolset[ScientistDeps] = FunctionToolset(id="scientist")
        self._memory_backend = self._init_memory_backend()
        self._register_tools()
        self._agent = self._create_agent()
        register_agent("scientist", self._agent)
        self._setup_memory()

    @property
    def agent(self) -> Agent[ScientistDeps, ResearchReport]:
        """Return the underlying pydantic-ai Agent."""
        return self._agent

    @property
    def settings(self) -> ScientistSettings:
        """Return current settings."""
        return self._settings

    # =========================================================================
    # Tool Methods
    # =========================================================================

    async def analyze_leaderboard(
        self,
        ctx: RunContext[ScientistDeps],
        refresh: bool = True,
    ) -> dict[str, Any]:
        """Analyze the current competition leaderboard."""
        with logfire.span("scientist.analyze_leaderboard"):
            if refresh:
                await ctx.deps.refresh_leaderboard()

            leaderboard = ctx.deps.leaderboard
            if not leaderboard:
                return {"error": "No leaderboard data available"}

            scores = [e.score for e in leaderboard]
            return {
                "total_teams": len(leaderboard),
                "top_score": max(scores),
                "median_score": sorted(scores)[len(scores) // 2],
                "score_range": max(scores) - min(scores),
                "top_10_scores": [e.score for e in leaderboard[:10]],
                "top_teams": [
                    {"rank": e.rank, "team": e.team_name, "score": e.score}
                    for e in leaderboard[:10]
                ],
            }

    async def get_kaggle_notebooks(
        self,
        ctx: RunContext[ScientistDeps],
        sort_by: str = "voteCount",
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top notebooks for the competition."""
        with logfire.span("scientist.get_notebooks"):
            notebooks = await self._fetch_kaggle_notebooks(
                ctx,
                sort_by=sort_by,
                max_results=max_results,
            )
            if notebooks:
                return notebooks

            await ctx.deps.refresh_leaderboard()
            fallback = []
            for entry in ctx.deps.leaderboard[:max_results]:
                fallback.append(
                    {
                        "title": f"{ctx.deps.competition.title} solution by {entry.team_name}",
                        "votes": max(1, (len(ctx.deps.leaderboard) - entry.rank + 1) * 5),
                        "author": entry.team_name,
                        "techniques": self._infer_techniques_from_text(
                            " ".join(ctx.deps.competition.tags)
                        ),
                    }
                )
            return fallback

    async def analyze_data_characteristics(
        self,
        ctx: RunContext[ScientistDeps],
    ) -> dict[str, Any]:
        """Analyze competition data characteristics."""
        with logfire.span("scientist.analyze_data"):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    files = await ctx.deps.platform_adapter.download_data(
                        ctx.deps.competition.id,
                        tmp_dir,
                    )
                    return self._summarize_dataset(files)
            except Exception as exc:
                logfire.warning("data_analysis_failed", error=str(exc))
                return self._fallback_dataset_summary(ctx.deps.competition)

    async def compute_baseline_estimate(
        self,
        ctx: RunContext[ScientistDeps],
        leaderboard_scores: list[float],
        competition_difficulty: str,
    ) -> float:
        """Estimate achievable baseline score."""
        _ = ctx
        if not leaderboard_scores:
            return 0.0

        median = sorted(leaderboard_scores)[len(leaderboard_scores) // 2]
        difficulty_multiplier = {
            "easy": 0.95,
            "medium": 0.85,
            "hard": 0.70,
        }.get(competition_difficulty, 0.80)

        return median * difficulty_multiplier

    def _init_memory_backend(self) -> AgentKMemoryTool | None:
        try:
            return create_memory_backend()
        except RuntimeError:  # pragma: no cover - optional dependency
            return None

    def _create_agent(self) -> Agent[ScientistDeps, ResearchReport]:
        """Create the underlying pydantic-ai agent."""
        builtin_tools: list[Any] = [prepare_web_search]
        if self._memory_backend is not None:
            builtin_tools.append(prepare_memory_tool)

        agent: Agent[ScientistDeps, ResearchReport] = Agent(
            model=get_model(self._settings.model),
            deps_type=ScientistDeps,
            output_type=ResearchReport,
            instructions=SCIENTIST_SYSTEM_PROMPT,
            name="scientist",
            model_settings=self._settings.model_settings,
            retries=self._settings.tool_retries,
            output_retries=self._settings.output_retries,
            builtin_tools=builtin_tools,
            toolsets=[
                create_production_toolset(
                    [self._toolset, cast("FunctionToolset[ScientistDeps]", kaggle_toolset)]
                ),
            ],
            prepare_tools=universal_tool_preparation,
            instrument=True,
        )

        agent.output_validator(self._validate_research_completeness)
        agent.instructions(self._add_competition_context)

        return agent

    def _setup_memory(self) -> None:
        """Set up memory tool if available."""
        if self._memory_backend is None:
            return
        register_memory_tool(self._agent, self._memory_backend)

    def _register_tools(self) -> None:
        """Register all research tools with the toolset."""
        self._toolset.tool(self.analyze_leaderboard)
        self._toolset.tool(self.get_kaggle_notebooks)
        self._toolset.tool(self.analyze_data_characteristics)
        self._toolset.tool(self.compute_baseline_estimate)

    async def _validate_research_completeness(
        self,
        ctx: RunContext[ScientistDeps],
        output: ResearchReport,
    ) -> ResearchReport:
        """Validate research report completeness."""
        if ctx.partial_output:
            return output
        if not output.recommended_approaches:
            raise ModelRetry("Research must include recommended approaches.")
        if not output.domain_findings and not output.technique_findings:
            raise ModelRetry("Research must include at least one finding.")
        return output

    async def _add_competition_context(self, ctx: RunContext[ScientistDeps]) -> str:
        """Add competition-specific context to instructions."""
        comp = ctx.deps.competition
        prize = f"${comp.prize_pool:,}" if comp.prize_pool else "N/A"
        tags = ", ".join(comp.tags) if comp.tags else "None"
        return (
            "CURRENT COMPETITION:\n"
            f"- ID: {comp.id}\n"
            f"- Title: {comp.title}\n"
            f"- Type: {comp.competition_type.value}\n"
            f"- Metric: {comp.metric.value} ({comp.metric_direction})\n"
            f"- Days Remaining: {comp.days_remaining}\n"
            f"- Prize Pool: {prize}\n"
            f"- Tags: {tags}"
        )

    async def _fetch_kaggle_notebooks(
        self,
        ctx: RunContext[ScientistDeps],
        *,
        sort_by: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {
            "competition": ctx.deps.competition.id,
            "sortBy": sort_by,
            "pageSize": max_results,
        }

        auth = self._extract_kaggle_auth(ctx.deps.platform_adapter)
        if not auth:
            return []

        response = await ctx.deps.http_client.get(
            _KAGGLE_KERNELS_ENDPOINT,
            params=params,
            auth=auth,
        )

        if response.status_code != 200:
            return []

        notebooks = []
        for item in response.json():
            title = item.get("title", "")
            notebooks.append(
                {
                    "title": title,
                    "votes": item.get("voteCount", 0),
                    "author": item.get("author", ""),
                    "techniques": self._infer_techniques_from_text(
                        f"{title} {item.get('scriptVersionTitle', '')}"
                    ),
                    "url": item.get("url", ""),
                }
            )

        return notebooks

    def _extract_kaggle_auth(self, adapter: PlatformAdapter) -> tuple[str, str] | None:
        if not hasattr(adapter, "config"):
            return None
        config = adapter.config
        username = getattr(config, "username", None)
        api_key = getattr(config, "api_key", None)
        if not username or not api_key:
            return None
        return (username, api_key)

    def _infer_techniques_from_text(self, text: str) -> list[str]:
        lower_text = text.lower()
        techniques = []
        for keyword, technique in _DEFAULT_NOTEBOOK_TECHNIQUES.items():
            if keyword in lower_text and technique not in techniques:
                techniques.append(technique)
        return techniques

    def _summarize_dataset(self, files: list[str]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "files": [],
            "total_size_mb": 0.0,
        }

        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                continue

            file_info: dict[str, Any] = {
                "name": path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }

            summary["total_size_mb"] += file_info["size_mb"]

            if path.suffix.lower() == ".csv":
                file_info.update(self._summarize_csv(path))

            summary["files"].append(file_info)

        summary["total_size_mb"] = round(summary["total_size_mb"], 2)
        return summary

    def _summarize_csv(self, path: Path) -> dict[str, Any]:
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        if not rows:
            return {"row_count": 0, "column_count": 0}

        header = rows[0]
        sample_rows = rows[1:101]
        missing_counts = {col: 0 for col in header}

        for row in sample_rows:
            for col, value in zip(header, row, strict=False):
                if value.strip().lower() in _MISSING_VALUE_TOKENS:
                    missing_counts[col] += 1

        missing_summary = {col: count for col, count in missing_counts.items() if count > 0}

        return {
            "row_count": len(rows) - 1,
            "column_count": len(header),
            "columns": header,
            "missing_values": missing_summary,
        }

    def _fallback_dataset_summary(self, competition: Competition) -> dict[str, Any]:
        return {
            "files": [],
            "total_size_mb": 0.0,
            "notes": f"Dataset summary unavailable for {competition.title}",
        }


# Module-level singleton for backward compatibility
scientist_agent_instance = ScientistAgent()
scientist_agent = scientist_agent_instance.agent
