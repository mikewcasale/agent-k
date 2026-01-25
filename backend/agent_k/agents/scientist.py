"""Scientist agent - research and analysis for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import csv
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

# Third-party (alphabetical)
import httpx
import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.agents import register_agent
from agent_k.agents.base import MemoryMixin, universal_tool_preparation
from agent_k.agents.prompts import SCIENTIST_SYSTEM_PROMPT
from agent_k.core.constants import DEFAULT_MODEL
from agent_k.core.data import locate_data_files
from agent_k.core.hints import DatasetProfile, build_dataset_profile, generate_preprocessing_hints
from agent_k.infra.providers import get_model
from agent_k.toolsets import (
    create_production_toolset,
    kaggle_toolset,
    prepare_memory_tool,
    prepare_web_fetch,
    prepare_web_search,
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
    "get_external_data_policy",
    "scientist_agent",
)

SCHEMA_VERSION: Final[str] = "1.0.0"
_KAGGLE_KERNELS_ENDPOINT: Final[str] = "https://www.kaggle.com/api/v1/kernels/list"
_KAGGLE_KERNEL_VIEW_ENDPOINT: Final[str] = "https://www.kaggle.com/api/v1/kernels/view"
_KAGGLE_KERNEL_CODE_KEYS: Final[tuple[str, ...]] = ("script", "code", "content")
_KAGGLE_RULES_ENDPOINT: Final[str] = "https://www.kaggle.com/competitions/{competition_id}/rules"
_EXTERNAL_DATA_ALLOW_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"external data.*(allowed|permitted|may be used)", re.IGNORECASE),
    re.compile(r"(outside|external|third[- ]party) data.*(allowed|permitted)", re.IGNORECASE),
    re.compile(r"you may use external data", re.IGNORECASE),
)
_EXTERNAL_DATA_BLOCK_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"no external data", re.IGNORECASE),
    re.compile(r"external data is not allowed", re.IGNORECASE),
    re.compile(r"do not use external data", re.IGNORECASE),
    re.compile(r"(outside|external) data.*(not allowed|prohibited|disallowed)", re.IGNORECASE),
)
_EXTERNAL_DATA_TERMS: Final[tuple[str, ...]] = (
    "external data",
    "outside data",
    "third-party data",
    "third party data",
    "additional data",
)
_ENRICHMENT_GEO_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?:^|[_\W])(?:lat|lon|latitude|longitude|zip|postal|postcode|address|neighborhood|district|region|city|state)(?:$|[_\W])",
        re.IGNORECASE,
    ),
)
_ENRICHMENT_PRICE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?:^|[_\W])(?:price|cost|value|amount|income|salary)(?:$|[_\W])", re.IGNORECASE),
)
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
    "blend": "stacking",
}
_KERNEL_TECHNIQUE_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "lightgbm": re.compile(r"\b(LGBMRegressor|LGBMClassifier|lightgbm|lgbm)\b", re.IGNORECASE),
    "xgboost": re.compile(r"\b(XGBRegressor|XGBClassifier|xgboost)\b", re.IGNORECASE),
    "catboost": re.compile(r"\b(CatBoostRegressor|CatBoostClassifier|catboost)\b", re.IGNORECASE),
    "stacking": re.compile(r"\bStacking(Regressor|Classifier)|stacking\b", re.IGNORECASE),
    "blending": re.compile(r"\bblend|blending|averaging|weighted average\b", re.IGNORECASE),
    "log1p": re.compile(r"\blog1p\b|\bnp\.log1p\b", re.IGNORECASE),
    "boxcox": re.compile(r"box[-_ ]?cox", re.IGNORECASE),
    "yeojohnson": re.compile(r"yeo[-_ ]?johnson", re.IGNORECASE),
    "power_transform": re.compile(r"\bPowerTransformer\b", re.IGNORECASE),
    "quantile_transform": re.compile(r"\bQuantileTransformer\b", re.IGNORECASE),
    "polynomial_features": re.compile(r"\bPolynomialFeatures\b", re.IGNORECASE),
    "binning": re.compile(r"\bKBinsDiscretizer\b|\bpd\.cut\b|\bpd\.qcut\b", re.IGNORECASE),
    "target_encoding": re.compile(r"\bTargetEncoder\b", re.IGNORECASE),
    "feature_scaling": re.compile(r"\b(StandardScaler|MinMaxScaler|RobustScaler)\b", re.IGNORECASE),
    "cross_validation": re.compile(r"\b(KFold|StratifiedKFold|cross_val_score|cross_validate)\b", re.IGNORECASE),
}
_MISSING_VALUE_TOKENS: Final[frozenset[str]] = frozenset({"", "na", "nan", "null", "none"})


class ScientistSettings(BaseSettings):
    """Configuration for the Scientist agent."""

    model_config = SettingsConfigDict(env_prefix="SCIENTIST_", env_file=".env", extra="ignore", validate_default=True)
    model: str = Field(default=DEFAULT_MODEL, description="Model identifier for research tasks")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature for research prompts")
    max_tokens: int = Field(default=4096, ge=1, description="Maximum tokens for responses")
    tool_retries: int = Field(default=2, ge=0, description="Tool retry attempts")
    output_retries: int = Field(default=1, ge=0, description="Output validation retry attempts")
    max_paper_results: int = Field(default=10, ge=1, description="Maximum papers to retrieve")
    max_notebook_results: int = Field(default=10, ge=1, description="Maximum notebooks to retrieve")

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings for the configured model."""
        return ModelSettings(temperature=self.temperature, max_tokens=self.max_tokens)


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
        self.leaderboard = await self.platform_adapter.get_leaderboard(self.competition.id, limit=100)


class ResearchFinding(BaseModel):
    """Individual research finding."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    category: str = Field(description="Category of finding")
    title: str = Field(description="Brief title")
    summary: str = Field(description="Detailed summary")
    relevance_score: float = Field(ge=0, le=1, description="Relevance to competition")
    sources: list[str] = Field(default_factory=list, description="Source URLs")


class LeaderboardAnalysis(BaseModel):
    """Analysis of competition leaderboard."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    top_score: float = Field(description="Best leaderboard score")
    median_score: float = Field(description="Median leaderboard score")
    score_distribution: str = Field(description="Description of score distribution")
    common_approaches: list[str] = Field(description="Inferred common approaches")
    improvement_opportunities: list[str] = Field(description="Potential improvement areas")


class ResearchReport(BaseModel):
    """Complete research report for a competition."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    competition_id: str = Field(description="Competition identifier")
    domain_findings: list[ResearchFinding] = Field(
        default_factory=list, description="Domain-specific research findings"
    )
    technique_findings: list[ResearchFinding] = Field(
        default_factory=list, description="Technique-focused research findings"
    )
    leaderboard_analysis: LeaderboardAnalysis | None = Field(default=None, description="Leaderboard analysis summary")
    recommended_approaches: list[str] = Field(default_factory=list, description="Recommended modeling approaches")
    estimated_baseline_score: float | None = Field(default=None, description="Estimated baseline score")
    key_challenges: list[str] = Field(default_factory=list, description="Primary competition challenges")


class ScientistAgent(MemoryMixin):
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

    async def analyze_leaderboard(self, ctx: RunContext[ScientistDeps], refresh: bool = True) -> dict[str, Any]:
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
                "top_teams": [{"rank": e.rank, "team": e.team_name, "score": e.score} for e in leaderboard[:10]],
            }

    async def get_kaggle_notebooks(
        self, ctx: RunContext[ScientistDeps], sort_by: str = "voteCount", max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Get top notebooks for the competition."""
        with logfire.span("scientist.get_notebooks"):
            notebooks = await self._fetch_kaggle_notebooks(ctx, sort_by=sort_by, max_results=max_results)
            if notebooks:
                return notebooks

            await ctx.deps.refresh_leaderboard()
            return [
                {
                    "title": f"{ctx.deps.competition.title} solution by {entry.team_name}",
                    "votes": max(1, (len(ctx.deps.leaderboard) - entry.rank + 1) * 5),
                    "author": entry.team_name,
                    "techniques": self._infer_techniques_from_text(" ".join(ctx.deps.competition.tags)),
                }
                for entry in ctx.deps.leaderboard[:max_results]
            ]

    async def analyze_top_kernels(
        self, ctx: RunContext[ScientistDeps], sort_by: str = "voteCount", max_results: int = 5
    ) -> dict[str, Any]:
        """Analyze top Kaggle kernels for techniques and patterns."""
        with logfire.span("scientist.analyze_top_kernels"):
            notebooks = await self._fetch_kaggle_notebooks(ctx, sort_by=sort_by, max_results=max_results)
            analyses: list[dict[str, Any]] = []
            for notebook in notebooks[:max_results]:
                kernel_ref = notebook.get("ref") or self._extract_kernel_ref(notebook.get("url", ""))
                code = await self._fetch_kernel_code(ctx, kernel_ref) if kernel_ref else None
                extracted = (
                    self._extract_techniques_from_code(code)
                    if code
                    else {"techniques": notebook.get("techniques", []), "target_transforms": [], "stacking": []}
                )
                analyses.append(
                    {
                        "title": notebook.get("title"),
                        "author": notebook.get("author"),
                        "votes": notebook.get("votes"),
                        "url": notebook.get("url"),
                        "ref": kernel_ref,
                        "techniques": extracted["techniques"],
                        "target_transforms": extracted["target_transforms"],
                        "stacking": extracted["stacking"],
                        "code_available": bool(code),
                    }
                )
            return {"kernels": analyses, "summary": self._summarize_kernel_analysis(analyses)}

    async def extract_techniques(self, ctx: RunContext[ScientistDeps], kernel_code: str) -> dict[str, Any]:
        """Extract modeling techniques from Kaggle kernel code."""
        _ = ctx
        return self._extract_techniques_from_code(kernel_code)

    async def synthesize_strategy(
        self,
        ctx: RunContext[ScientistDeps],
        techniques: list[str] | None = None,
        target_transforms: list[str] | None = None,
        stacking: list[str] | None = None,
    ) -> dict[str, Any]:
        """Synthesize a prioritized strategy from extracted techniques."""
        _ = ctx
        techniques = techniques or []
        target_transforms = target_transforms or []
        stacking = stacking or []
        plan: list[dict[str, Any]] = []
        priority = 1

        if target_transforms:
            plan.append(
                {
                    "priority": priority,
                    "action": "target_transform",
                    "details": f"Evaluate {', '.join(sorted(set(target_transforms)))} for the target.",
                }
            )
            priority += 1

        if stacking or "stacking" in techniques or "blending" in techniques:
            plan.append(
                {
                    "priority": priority,
                    "action": "stacking_blending",
                    "details": "Prototype a simple stacking/blending ensemble with diverse base models.",
                }
            )
            priority += 1

        if "feature_scaling" in techniques or "polynomial_features" in techniques or "binning" in techniques:
            plan.append(
                {
                    "priority": priority,
                    "action": "feature_engineering",
                    "details": "Prioritize scaling, polynomial interactions, and binning for numeric features.",
                }
            )
            priority += 1

        if "lightgbm" in techniques:
            plan.append(
                {
                    "priority": priority,
                    "action": "lightgbm_objectives",
                    "details": "Tune LightGBM objectives with quantile or huber settings for robustness.",
                }
            )
            priority += 1

        return {"plan": plan, "signals": {"techniques": techniques, "stacking": stacking}}

    async def analyze_data_characteristics(self, ctx: RunContext[ScientistDeps]) -> dict[str, Any]:
        """Analyze competition data characteristics."""
        with logfire.span("scientist.analyze_data"):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    files = await ctx.deps.platform_adapter.download_data(ctx.deps.competition.id, tmp_dir)
                    summary = self._summarize_dataset(files)
                    profile: DatasetProfile | None = None
                    try:
                        train_path, test_path, sample_path = locate_data_files(files)
                        profile = build_dataset_profile(train_path, test_path, sample_path)
                        hints = generate_preprocessing_hints(profile, ctx.deps.competition.id)
                        summary["dataset_profile"] = profile.to_dict()
                        summary["preprocessing_hints"] = [hint.to_dict() for hint in hints]
                    except Exception as exc:
                        logfire.warning("dataset_profile_failed", error=str(exc))
                    try:
                        external_data = await get_external_data_policy(
                            ctx.deps.http_client,
                            ctx.deps.competition.id,
                            profile=profile,
                            cache=ctx.deps.research_cache,
                        )
                        summary["external_data_rules"] = external_data
                    except Exception as exc:
                        logfire.warning("external_data_rules_failed", error=str(exc))
                    return summary
            except Exception as exc:
                logfire.warning("data_analysis_failed", error=str(exc))
                return self._fallback_dataset_summary(ctx.deps.competition)

    async def check_external_data_rules(
        self, ctx: RunContext[ScientistDeps], profile: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Check competition rules for external data usage."""
        dataset_profile = DatasetProfile.from_dict(profile) if isinstance(profile, dict) else None
        return await get_external_data_policy(
            ctx.deps.http_client, ctx.deps.competition.id, profile=dataset_profile, cache=ctx.deps.research_cache
        )

    async def compute_baseline_estimate(
        self, ctx: RunContext[ScientistDeps], leaderboard_scores: list[float], competition_difficulty: str
    ) -> float:
        """Estimate achievable baseline score."""
        _ = ctx
        if not leaderboard_scores:
            return 0.0

        median = sorted(leaderboard_scores)[len(leaderboard_scores) // 2]
        difficulty_multiplier = {"easy": 0.95, "medium": 0.85, "hard": 0.70}.get(competition_difficulty, 0.80)

        return median * difficulty_multiplier

    def _create_agent(self) -> Agent[ScientistDeps, ResearchReport]:
        """Create the underlying pydantic-ai agent."""
        builtin_tools: list[Any] = [prepare_web_search, prepare_web_fetch]
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
                create_production_toolset([self._toolset, cast("FunctionToolset[ScientistDeps]", kaggle_toolset)])
            ],
            prepare_tools=universal_tool_preparation,
            instrument=True,
        )

        agent.output_validator(self._validate_research_completeness)
        agent.instructions(self._add_competition_context)

        return agent

    def _register_tools(self) -> None:
        """Register all research tools with the toolset."""
        self._toolset.tool(self.analyze_leaderboard)
        self._toolset.tool(self.get_kaggle_notebooks)
        self._toolset.tool(self.analyze_top_kernels)
        self._toolset.tool(self.extract_techniques)
        self._toolset.tool(self.synthesize_strategy)
        self._toolset.tool(self.analyze_data_characteristics)
        self._toolset.tool(self.check_external_data_rules)
        self._toolset.tool(self.compute_baseline_estimate)

    async def _validate_research_completeness(
        self, ctx: RunContext[ScientistDeps], output: ResearchReport
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
        self, ctx: RunContext[ScientistDeps], *, sort_by: str, max_results: int
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {
            "competition": ctx.deps.competition.id,
            "sortBy": sort_by,
            "pageSize": max_results,
        }

        auth = self._extract_kaggle_auth(ctx.deps.platform_adapter)
        if not auth:
            return []

        response = await ctx.deps.http_client.get(_KAGGLE_KERNELS_ENDPOINT, params=params, auth=auth)
        if response.status_code != 200:
            return []

        results: list[dict[str, Any]] = []
        for item in response.json():
            ref = item.get("ref") or item.get("id") or item.get("kernelId")
            results.append(
                {
                    "title": item.get("title", ""),
                    "votes": item.get("voteCount", 0),
                    "author": item.get("author", ""),
                    "techniques": self._infer_techniques_from_text(
                        f"{item.get('title', '')} {item.get('scriptVersionTitle', '')}"
                    ),
                    "url": item.get("url", ""),
                    "ref": ref,
                }
            )
        return results

    async def _fetch_kernel_code(self, ctx: RunContext[ScientistDeps], kernel_ref: str) -> str | None:
        auth = self._extract_kaggle_auth(ctx.deps.platform_adapter)
        if not auth:
            return None
        for key in ("kernelId", "kernelSlug"):
            response = await ctx.deps.http_client.get(_KAGGLE_KERNEL_VIEW_ENDPOINT, params={key: kernel_ref}, auth=auth)
            if response.status_code != 200:
                continue
            payload = response.json()
            for code_key in _KAGGLE_KERNEL_CODE_KEYS:
                value = payload.get(code_key)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _extract_kernel_ref(self, url: str) -> str | None:
        if not url:
            return None
        match = re.search(r"kaggle\\.com/(?:code/)?(?P<owner>[^/]+)/(?P<slug>[^/?#]+)", url)
        if not match:
            return None
        return f"{match.group('owner')}/{match.group('slug')}"

    def _extract_techniques_from_code(self, kernel_code: str) -> dict[str, Any]:
        techniques: list[str] = []
        target_transforms: list[str] = []
        stacking: list[str] = []
        for name, pattern in _KERNEL_TECHNIQUE_PATTERNS.items():
            if not kernel_code or not pattern.search(kernel_code):
                continue
            if name in {"log1p", "boxcox", "yeojohnson"}:
                target_transforms.append(name)
            elif name in {"stacking", "blending"}:
                stacking.append(name)
            else:
                techniques.append(name)
        return {
            "techniques": sorted(set(techniques)),
            "target_transforms": sorted(set(target_transforms)),
            "stacking": sorted(set(stacking)),
        }

    def _summarize_kernel_analysis(self, analyses: list[dict[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        targets: dict[str, int] = {}
        for entry in analyses:
            for technique in entry.get("techniques", []):
                counts[technique] = counts.get(technique, 0) + 1
            for transform in entry.get("target_transforms", []):
                targets[transform] = targets.get(transform, 0) + 1
        return {"technique_counts": counts, "target_transform_counts": targets, "total_kernels": len(analyses)}

    def _extract_kaggle_auth(self, adapter: PlatformAdapter) -> tuple[str, str] | None:
        if not hasattr(adapter, "config"):
            return None
        config = adapter.config
        username, api_key = getattr(config, "username", None), getattr(config, "api_key", None)
        return (username, api_key) if username and api_key else None

    def _infer_techniques_from_text(self, text: str) -> list[str]:
        lower_text = text.lower()
        techniques = []
        for keyword, technique in _DEFAULT_NOTEBOOK_TECHNIQUES.items():
            if keyword in lower_text and technique not in techniques:
                techniques.append(technique)
        return techniques

    def _summarize_dataset(self, files: list[str]) -> dict[str, Any]:
        summary: dict[str, Any] = {"files": [], "total_size_mb": 0.0}

        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                continue

            file_info: dict[str, Any] = {"name": path.name, "size_mb": round(path.stat().st_size / (1024 * 1024), 2)}
            summary["total_size_mb"] += file_info["size_mb"]
            if path.suffix.lower() == ".csv":
                file_info.update(self._summarize_csv(path))

            summary["files"].append(file_info)

        summary["total_size_mb"] = round(summary["total_size_mb"], 2)
        return summary

    def _summarize_csv(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8", newline="") as handle:
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

        return {
            "row_count": len(rows) - 1,
            "column_count": len(header),
            "columns": header,
            "missing_values": {col: count for col, count in missing_counts.items() if count > 0},
        }

    def _fallback_dataset_summary(self, competition: Competition) -> dict[str, Any]:
        return {"files": [], "total_size_mb": 0.0, "notes": f"Dataset summary unavailable for {competition.title}"}


async def get_external_data_policy(
    http_client: httpx.AsyncClient,
    competition_id: str,
    *,
    profile: DatasetProfile | None = None,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch and parse competition rules for external data usage."""
    cache_key = f"external_data_policy:{competition_id}"
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)

    rules_url = _KAGGLE_RULES_ENDPOINT.format(competition_id=competition_id)
    rules_text = ""
    try:
        response = await http_client.get(rules_url)
        if response.status_code == 200:
            rules_text = response.text
    except httpx.HTTPError as exc:
        logfire.warning("external_data_rules_fetch_failed", error=str(exc), competition_id=competition_id)

    allowed, restrictions = _parse_external_data_policy(rules_text)
    recommended = _suggest_enrichment_sources(profile) if allowed else []
    payload = {
        "external_data_allowed": allowed,
        "restrictions": restrictions,
        "recommended_sources": recommended,
        "rules_url": rules_url,
    }
    if cache is not None:
        cache[cache_key] = payload
    return payload


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_external_data_policy(rules_text: str) -> tuple[bool | None, list[str]]:
    if not rules_text:
        return None, []
    cleaned = _strip_html(rules_text)
    if not cleaned:
        return None, []
    allowed: bool | None = None
    if any(pattern.search(cleaned) for pattern in _EXTERNAL_DATA_BLOCK_PATTERNS):
        allowed = False
    elif any(pattern.search(cleaned) for pattern in _EXTERNAL_DATA_ALLOW_PATTERNS):
        allowed = True
    restrictions = _extract_external_data_restrictions(cleaned)
    return allowed, restrictions


def _extract_external_data_restrictions(text: str, *, limit: int = 5) -> list[str]:
    sentences = re.split(r"[.!?]\s+", text)
    results: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in _EXTERNAL_DATA_TERMS):
            cleaned = sentence.strip()
            if cleaned:
                results.append(cleaned)
        if len(results) >= limit:
            break
    return results


def _suggest_enrichment_sources(profile: DatasetProfile | None) -> list[str]:
    if profile is None:
        return []
    names = list(profile.columns)
    has_geo = profile.has_geographic_features or any(
        pattern.search(name) for name in names for pattern in _ENRICHMENT_GEO_PATTERNS
    )
    has_temporal = profile.has_temporal_features
    has_price = profile.has_price_features or any(
        pattern.search(name) for name in names for pattern in _ENRICHMENT_PRICE_PATTERNS
    )

    suggestions: list[str] = []
    if has_geo:
        suggestions.extend(
            [
                "Census demographic statistics (national or regional datasets)",
                "Geographic boundaries or zoning layers (e.g., OpenStreetMap)",
                "Points of interest density or amenity proximity",
            ]
        )
    if has_temporal:
        suggestions.extend(
            ["Economic indicators aligned to date (macro trends, rates)", "Holiday or seasonal calendars"]
        )
    if has_price:
        suggestions.extend(["Inflation indices or consumer price series", "Market index or sector pricing benchmarks"])
    return suggestions


# Module-level singleton for backward compatibility
scientist_agent_instance = ScientistAgent()
scientist_agent = scientist_agent_instance.agent
