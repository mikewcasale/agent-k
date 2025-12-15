"""Scientist agent for research and analysis.

The Scientist agent performs literature review, leaderboard analysis,
and domain research to inform solution strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.builtin_tools import WebSearchTool

from ...core.constants import DEFAULT_MODEL
from ...core.models import Competition, LeaderboardEntry
from ...core.protocols import PlatformAdapter
from ...infra.models import get_model

__all__ = [
    'ScientistDeps',
    'ResearchFinding',
    'LeaderboardAnalysis',
    'ResearchReport',
    'create_scientist_agent',
    'ScientistAgent',
]


# =============================================================================
# Dependency Container
# =============================================================================
@dataclass
class ScientistDeps:
    """Dependencies for the Scientist agent.
    
    The dependency container holds all external services required by the agent.
    Using a dataclass ensures type safety and clear documentation of requirements.
    
    Attributes:
        http_client: Async HTTP client for web requests.
        platform_adapter: Platform adapter for competition data.
        competition: Target competition being researched.
        leaderboard: Current leaderboard state (refreshed during research).
        research_cache: Cache for expensive research operations.
    """
    
    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    competition: Competition
    leaderboard: list[LeaderboardEntry] = field(default_factory=list)
    research_cache: dict[str, Any] = field(default_factory=dict)
    
    async def refresh_leaderboard(self) -> None:
        """Refresh leaderboard from platform."""
        self.leaderboard = await self.platform_adapter.get_leaderboard(
            self.competition.id,
            limit=100,
        )


# =============================================================================
# Output Models
# =============================================================================
class ResearchFinding(BaseModel):
    """Individual research finding."""
    
    category: str = Field(description='Category of finding')
    title: str = Field(description='Brief title')
    summary: str = Field(description='Detailed summary')
    relevance_score: float = Field(ge=0, le=1, description='Relevance to competition')
    sources: list[str] = Field(default_factory=list, description='Source URLs')


class LeaderboardAnalysis(BaseModel):
    """Analysis of competition leaderboard."""
    
    top_score: float
    median_score: float
    score_distribution: str = Field(description='Description of score distribution')
    common_approaches: list[str] = Field(description='Inferred common approaches')
    improvement_opportunities: list[str] = Field(description='Potential improvement areas')


class ResearchReport(BaseModel):
    """Complete research report for a competition."""
    
    competition_id: str
    domain_findings: list[ResearchFinding] = Field(default_factory=list)
    technique_findings: list[ResearchFinding] = Field(default_factory=list)
    leaderboard_analysis: LeaderboardAnalysis | None = None
    recommended_approaches: list[str] = Field(default_factory=list)
    estimated_baseline_score: float | None = None
    key_challenges: list[str] = Field(default_factory=list)


# =============================================================================
# Agent Definition
# =============================================================================
def create_scientist_agent(model: str = DEFAULT_MODEL) -> Agent[ScientistDeps, ResearchReport]:
    """Create and configure the Scientist agent.
    
    The Scientist agent performs comprehensive research including:
    - Academic literature review via web search
    - Leaderboard analysis and pattern detection
    - Domain-specific best practice identification
    - Solution approach recommendations
    
    Args:
        model: Model specification string. Supports:
            - Standard pydantic-ai strings: 'anthropic:claude-sonnet-4-5'
            - Devstral local: 'devstral:local'
            - Devstral with custom URL: 'devstral:http://localhost:1234/v1'
    
    Returns:
        Configured Scientist agent.
    """
    # Resolve model specification to Model instance or string
    resolved_model = get_model(model)
    
    agent = Agent(
        resolved_model,
        deps_type=ScientistDeps,
        output_type=ResearchReport,
        instructions=_get_scientist_instructions(),
        builtin_tools=[WebSearchTool()],
        retries=2,
    )
    
    # =========================================================================
    # Dynamic Instructions (Dependency-Aware)
    # =========================================================================
    @agent.instructions
    async def add_competition_context(ctx: RunContext[ScientistDeps]) -> str:
        """Add competition-specific context to instructions."""
        comp = ctx.deps.competition
        return f"""
CURRENT COMPETITION:
- ID: {comp.id}
- Title: {comp.title}
- Type: {comp.competition_type.value}
- Metric: {comp.metric.value} ({comp.metric_direction})
- Days Remaining: {comp.days_remaining}
- Prize Pool: ${comp.prize_pool:,} if comp.prize_pool else 'N/A'
- Tags: {', '.join(comp.tags) if comp.tags else 'None'}
"""
    
    # =========================================================================
    # Tool Definitions
    # =========================================================================
    @agent.tool
    async def analyze_leaderboard(
        ctx: RunContext[ScientistDeps],
        refresh: bool = True,
    ) -> dict[str, Any]:
        """Analyze the current competition leaderboard.
        
        Args:
            ctx: Run context with dependencies.
            refresh: Whether to refresh leaderboard data first.
        
        Returns:
            Leaderboard statistics and analysis.
        """
        with logfire.span('scientist.analyze_leaderboard'):
            if refresh:
                await ctx.deps.refresh_leaderboard()
            
            leaderboard = ctx.deps.leaderboard
            if not leaderboard:
                return {'error': 'No leaderboard data available'}
            
            scores = [e.score for e in leaderboard]
            return {
                'total_teams': len(leaderboard),
                'top_score': max(scores),
                'median_score': sorted(scores)[len(scores) // 2],
                'score_range': max(scores) - min(scores),
                'top_10_scores': [e.score for e in leaderboard[:10]],
                'top_teams': [
                    {'rank': e.rank, 'team': e.team_name, 'score': e.score}
                    for e in leaderboard[:10]
                ],
            }
    
    @agent.tool
    async def search_papers(
        ctx: RunContext[ScientistDeps],
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, str]]:
        """Search for relevant academic papers.
        
        Args:
            ctx: Run context.
            query: Search query for papers.
            max_results: Maximum papers to return.
        
        Returns:
            List of paper metadata.
        """
        with logfire.span('scientist.search_papers', query=query):
            # Use HTTP client to search arXiv or similar
            response = await ctx.deps.http_client.get(
                'https://api.semanticscholar.org/graph/v1/paper/search',
                params={
                    'query': query,
                    'limit': max_results,
                    'fields': 'title,abstract,year,citationCount,url',
                },
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            return [
                {
                    'title': p.get('title', ''),
                    'abstract': p.get('abstract', '')[:500] if p.get('abstract') else '',
                    'year': str(p.get('year', '')),
                    'citations': str(p.get('citationCount', 0)),
                    'url': p.get('url', ''),
                }
                for p in data.get('data', [])
            ]
    
    @agent.tool
    async def get_kaggle_notebooks(
        ctx: RunContext[ScientistDeps],
        sort_by: str = 'voteCount',
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top notebooks for the competition.
        
        Args:
            ctx: Run context.
            sort_by: Sort criteria (voteCount, dateCreated).
            max_results: Maximum notebooks to return.
        
        Returns:
            List of notebook metadata.
        """
        with logfire.span('scientist.get_notebooks'):
            # This would use the Kaggle API via adapter
            # Placeholder implementation
            return [
                {
                    'title': 'EDA and Baseline',
                    'votes': 150,
                    'author': 'top_kaggler',
                    'techniques': ['lightgbm', 'feature_engineering', 'cross_validation'],
                }
            ]
    
    @agent.tool
    async def analyze_data_characteristics(
        ctx: RunContext[ScientistDeps],
    ) -> dict[str, Any]:
        """Analyze competition data characteristics.
        
        Returns information about the dataset structure, size, and properties.
        """
        with logfire.span('scientist.analyze_data'):
            # Would download and analyze actual data
            return {
                'data_type': 'tabular',
                'estimated_rows': 100000,
                'features': ['numerical', 'categorical'],
                'target_distribution': 'balanced',
                'missing_values': 'moderate',
            }
    
    @agent.tool_plain
    async def compute_baseline_estimate(
        leaderboard_scores: list[float],
        competition_difficulty: str,
    ) -> float:
        """Estimate achievable baseline score.
        
        Args:
            leaderboard_scores: Current leaderboard scores.
            competition_difficulty: Assessed difficulty level.
        
        Returns:
            Estimated achievable baseline score.
        """
        # Simple heuristic for baseline estimation
        if not leaderboard_scores:
            return 0.0
        
        median = sorted(leaderboard_scores)[len(leaderboard_scores) // 2]
        difficulty_multiplier = {
            'easy': 0.95,
            'medium': 0.85,
            'hard': 0.70,
        }.get(competition_difficulty, 0.80)
        
        return median * difficulty_multiplier
    
    # =========================================================================
    # Output Validators
    # =========================================================================
    @agent.output_validator
    async def validate_research_completeness(
        ctx: RunContext[ScientistDeps],
        output: ResearchReport,
    ) -> ResearchReport:
        """Validate research report completeness."""
        if not output.recommended_approaches:
            raise ValueError('Research must include recommended approaches')
        if not output.domain_findings and not output.technique_findings:
            raise ValueError('Research must include at least one finding')
        return output
    
    return agent


def _get_scientist_instructions() -> str:
    """Return system instructions for the Scientist agent."""
    return """You are the Scientist agent in the AGENT-K multi-agent system.

Your mission is to conduct comprehensive research for Kaggle competitions.

RESEARCH WORKFLOW:
1. Analyze the leaderboard to understand current performance landscape
2. Search academic papers for relevant techniques and approaches
3. Review top Kaggle notebooks for practical implementations
4. Analyze data characteristics to inform approach selection
5. Synthesize findings into actionable recommendations

RESEARCH PRIORITIES:
- Focus on techniques proven effective for similar competitions
- Identify quick wins for establishing a strong baseline
- Note advanced techniques for later optimization phases
- Consider computational constraints and time limits

OUTPUT REQUIREMENTS:
- Provide at least 3 recommended approaches with rationale
- Include estimated baseline score based on leaderboard analysis
- List key challenges that will need to be addressed
- Cite sources for all significant findings
"""


# =============================================================================
# Public Interface
# =============================================================================
class ScientistAgent:
    """High-level interface for the Scientist agent.
    
    Provides a clean API for research operations while encapsulating
    the underlying Pydantic-AI agent.
    
    Example:
        >>> agent = ScientistAgent()
        >>> async with httpx.AsyncClient() as client:
        ...     deps = ScientistDeps(
        ...         http_client=client,
        ...         platform_adapter=adapter,
        ...         competition=competition,
        ...     )
        ...     report = await agent.research(
        ...         'Conduct research for this tabular classification competition',
        ...         deps=deps,
        ...     )
    """
    
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._agent = create_scientist_agent(model)
    
    async def research(
        self,
        prompt: str,
        *,
        deps: ScientistDeps,
    ) -> ResearchReport:
        """Conduct research for a competition.
        
        Args:
            prompt: Research directive or focus areas.
            deps: Dependency container.
        
        Returns:
            Comprehensive research report.
        """
        with logfire.span('scientist.research', competition_id=deps.competition.id):
            result = await self._agent.run(prompt, deps=deps)
            return result.output
    
    async def quick_analysis(
        self,
        deps: ScientistDeps,
    ) -> LeaderboardAnalysis:
        """Perform quick leaderboard analysis without full research."""
        with logfire.span('scientist.quick_analysis'):
            await deps.refresh_leaderboard()
            
            scores = [e.score for e in deps.leaderboard]
            return LeaderboardAnalysis(
                top_score=max(scores) if scores else 0.0,
                median_score=sorted(scores)[len(scores) // 2] if scores else 0.0,
                score_distribution='Analysis pending',
                common_approaches=[],
                improvement_opportunities=[],
            )
