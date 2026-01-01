# SCIENTIST - Research Agent

The SCIENTIST agent conducts research on the selected competition. It analyzes leaderboards, searches for relevant papers and notebooks, and summarizes strategies.

## Role

- Analyze leaderboard score distribution
- Surface relevant notebooks and research
- Summarize dataset characteristics
- Recommend modeling approaches

## Tools

| Tool | Purpose |
|------|---------|
| `analyze_leaderboard` | Summarize leaderboard stats |
| `get_kaggle_notebooks` | Find top notebooks for the competition |
| `analyze_data_characteristics` | Inspect dataset structure |
| `compute_baseline_estimate` | Estimate a baseline score |
| `kaggle_*` toolset | Kaggle API helper tools |
| `web_search` (builtin) | Retrieve papers and discussions |
| `memory` (builtin) | Shared notes (Anthropic only) |

## Basic Usage

```python
from agent_k.agents.scientist import ScientistDeps, scientist_agent

competition = await kaggle_adapter.get_competition("titanic")

deps = ScientistDeps(
    http_client=http,
    platform_adapter=kaggle_adapter,
    competition=competition,
)

run_result = await scientist_agent.run(
    "Research the provided competition and summarize approaches",
    deps=deps,
)

output = run_result.output
print(output.recommended_approaches)
```

## Dependencies

```python
from dataclasses import dataclass, field
from typing import Any
import httpx

@dataclass
class ScientistDeps:
    """Dependencies for the SCIENTIST agent."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    competition: Competition
    leaderboard: list[LeaderboardEntry] = field(default_factory=list)
    research_cache: dict[str, Any] = field(default_factory=dict)
```

## Output Model

```python
class ResearchFinding(BaseModel):
    """Individual research finding."""

    category: str
    title: str
    summary: str
    relevance_score: float
    sources: list[str]

class LeaderboardAnalysis(BaseModel):
    """Leaderboard statistics summary."""

    top_score: float
    median_score: float
    score_distribution: str
    common_approaches: list[str]
    improvement_opportunities: list[str]

class ResearchReport(BaseModel):
    """Output from SCIENTIST research."""

    competition_id: str
    domain_findings: list[ResearchFinding]
    technique_findings: list[ResearchFinding]
    leaderboard_analysis: LeaderboardAnalysis | None
    recommended_approaches: list[str]
    estimated_baseline_score: float | None
    key_challenges: list[str]
```

## Notes

- The SCIENTIST output is converted into a simplified `ResearchFindings` object by the mission graph.
- The memory tool is only available for Anthropic models.
