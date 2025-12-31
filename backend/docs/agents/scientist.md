# SCIENTIST — Research Agent

The SCIENTIST agent conducts comprehensive research on the selected competition. It analyzes leaderboards, searches for relevant papers, and synthesizes a winning strategy.

## Role

- Analyze leaderboard score distribution
- Search for relevant academic papers
- Find winning approaches from past competitions
- Synthesize strategic recommendations
- Store findings for EVOLVER

## Tools

| Tool | Purpose |
|------|---------|
| `analyze_leaderboard` | Summarize leaderboard stats |
| `get_kaggle_notebooks` | Find top notebooks |
| `analyze_data_characteristics` | Inspect dataset structure |
| `compute_baseline_estimate` | Estimate baseline performance |
| `web_search` (builtin) | Retrieve papers and Kaggle discussions |
| `memory` (builtin) | Persist research notes (create/view) |

## Basic Usage

```python
from agent_k.agents.scientist import ScientistDeps, scientist_agent
from agent_k.ui.ag_ui import EventEmitter

# Create dependencies
competition = await kaggle_adapter.get_competition("titanic")
deps = ScientistDeps(
    http_client=http,
    platform_adapter=kaggle_adapter,
    competition=competition,
    event_emitter=EventEmitter(),
)

# Run research
run_result = await scientist_agent.run(
    prompt="""
    Research the provided competition.
    
    Analyze:
    1. Leaderboard score distribution
    2. Target score for top 10%
    3. Relevant papers and techniques
    4. Winning approaches from similar competitions
    
    Synthesize a strategy for achieving top 10%.
    """,
    deps=deps,
)

output = run_result.output
print(output.recommended_approaches)
```

## Research Process

```mermaid
graph TD
    A[Retrieve Competition] --> B[Get Leaderboard]
    B --> C[Analyze Distribution]
    C --> D[Calculate Target Score]
    D --> E[Search Papers]
    E --> F[Find Winning Solutions]
    F --> G[Synthesize Strategy]
    G --> H[Store Findings]
```

### 1. Retrieve Competition

```python
notes = await memory(command="view", path="shared/target_competition.md")
```

### 2. Get Leaderboard

```python
@toolset.tool
async def kaggle_get_leaderboard(
    competition_id: str,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Get competition leaderboard entries."""
    entries = await adapter.get_leaderboard(competition_id, limit=page_size)
    return [e.model_dump() for e in entries]
```

### 3. Analyze Distribution

Calculate statistics from leaderboard:

```python
leaderboard = await kaggle_get_leaderboard(comp_id)

scores = [entry["score"] for entry in leaderboard]
analysis = {
    "top_score": max(scores),
    "median_score": median(scores),
    "p90_score": percentile(scores, 90),
    "p10_score": percentile(scores, 10),
    "std_dev": std(scores),
}
```

### 4. Calculate Target Score

```python
def get_target_score(
    leaderboard: list[dict],
    target_percentile: float,
) -> float:
    """Calculate score needed for target percentile."""
    scores = sorted([e["score"] for e in leaderboard], reverse=True)
    target_rank = int(len(scores) * target_percentile)
    return scores[target_rank]
```

### 5. Search Papers

Use the built-in `web_search` tool with academic site filters:

```text
web_search(query="site:arxiv.org OR site:paperswithcode.com <topic>")
```

### 6. Find Winning Solutions

Use the built-in `web_search` tool scoped to Kaggle:

```text
web_search(query="site:kaggle.com <topic> winning solution")
```

### 7. Synthesize Strategy

The agent combines findings into actionable recommendations.

### 8. Store Findings

```python
await memory(
    command="create",
    path="shared/research_findings.md",
    file_text="...research summary...",
)
```

## Output Model

```python
class ResearchFindings(BaseModel):
    """Output from SCIENTIST research."""
    
    leaderboard_analysis: LeaderboardAnalysis
    papers: list[Paper]
    approaches: list[ApproachSummary]
    strategy_recommendations: list[str]
    key_techniques: list[str]
    potential_pitfalls: list[str]

class LeaderboardAnalysis(BaseModel):
    """Leaderboard statistics."""
    
    total_entries: int
    top_score: float
    median_score: float
    target_score: float
    target_percentile: float
    score_distribution: dict[str, float]
```

## Agent Instructions

```python
def get_scientist_instructions() -> str:
    return """You are the SCIENTIST agent in the AGENT-K system.

Your mission is to conduct comprehensive research on a Kaggle competition.

AVAILABLE TOOLS:
1. memory - Retrieve competition notes from LOBBYIST (view)
2. kaggle_get_leaderboard - Analyze current standings
3. web_search - Find academic papers and Kaggle discussions
4. memory - Save your findings (create)

WORKFLOW:
1. Retrieve the target competition: memory(view shared/target_competition.md)
2. Get and analyze the leaderboard
3. Calculate target score for user's percentile goal
4. Search for relevant papers and techniques (web_search with site filters)
5. Find winning approaches from similar competitions (web_search site:kaggle.com)
6. Synthesize strategy recommendations
7. Store findings via MemoryTool (create shared/research_findings.md)

ANALYSIS FOCUS:
- What separates top performers from average?
- What techniques are commonly used?
- What are common mistakes to avoid?
- What is the minimum viable solution?
- What innovations could provide edge?
"""
```

## Leaderboard Analysis

The SCIENTIST performs detailed leaderboard analysis:

```python
async def analyze_leaderboard(
    leaderboard: list[dict],
    target_percentile: float,
) -> LeaderboardAnalysis:
    scores = [e["score"] for e in leaderboard]
    
    return LeaderboardAnalysis(
        total_entries=len(scores),
        top_score=max(scores),
        median_score=median(scores),
        target_score=get_target_score(leaderboard, target_percentile),
        target_percentile=target_percentile,
        score_distribution={
            "p99": percentile(scores, 99),
            "p95": percentile(scores, 95),
            "p90": percentile(scores, 90),
            "p75": percentile(scores, 75),
            "p50": percentile(scores, 50),
            "p25": percentile(scores, 25),
        },
    )
```

## Paper Search

The SCIENTIST uses the built-in `web_search` tool for academic sources:

```text
web_search(query="site:arxiv.org OR site:paperswithcode.com Titanic survival prediction machine learning")
web_search(query="site:arxiv.org OR site:paperswithcode.com gradient boosting ensemble methods tabular data")
```

## Winning Solution Analysis

Use `web_search` scoped to Kaggle for winning solution discussions:

```text
web_search(query="site:kaggle.com Titanic winning solution discussion")
```

## Strategy Synthesis

The agent synthesizes findings into actionable strategy:

```
Based on research:

STRATEGY RECOMMENDATIONS:
1. Use ensemble of gradient boosting models (XGBoost + LightGBM)
2. Focus on feature engineering:
   - Family size features
   - Title extraction from names
   - Cabin deck features
3. Handle missing values:
   - Age: Impute with median by class
   - Embarked: Most frequent
4. Cross-validation strategy:
   - 5-fold stratified
   - Monitor for overfitting

KEY TECHNIQUES:
- Gradient boosting
- Feature engineering
- Ensemble methods

POTENTIAL PITFALLS:
- Overfitting to train set
- Ignoring class imbalance
- Leaking test information
```

## Integration with LYCURGUS

```python
@dataclass
class ResearchNode(BaseNode[MissionState, MissionResult]):
    scientist_agent: Agent
    
    async def run(self, ctx: ...) -> PrototypeNode | End[MissionResult]:
        # Build prompt
        prompt = f"""
        Research competition: {ctx.state.selected_competition.id}
        Target percentile: {ctx.state.criteria.target_leaderboard_percentile}
        
        Analyze leaderboard and develop strategy.
        """
        
        # Run SCIENTIST
        result = await self.scientist_agent.run(prompt, deps=deps)
        
        # Update state
        ctx.state.research_findings = result.data
        
        # Transition to Prototype
        return PrototypeNode(...)
```

## API Reference

See [API Reference: SCIENTIST](../api/agents/scientist.md) for complete documentation.
