# Multi-Agent Demo

This example demonstrates AGENT-K's multi-agent system with the LOBBYIST (discovery) and SCIENTIST (research) agents working together.

## Overview

The `multi_agent_playbook.py` script runs through two phases:

1. **Discovery**: LOBBYIST searches for Kaggle competitions
2. **Research**: SCIENTIST analyzes the selected competition

## Running the Demo

```bash
cd backend

# With local Devstral
uv run python examples/multi_agent_playbook.py --model devstral:local

# With Claude Haiku
uv run python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307

# With OpenRouter
uv run python examples/multi_agent_playbook.py --model openrouter:mistralai/devstral-small
```

## Code Walkthrough

### Imports and Setup

```python
import asyncio
from pathlib import Path

from agent_k.infra.models import get_model
from agent_k.toolsets import (
    create_kaggle_toolset,
    create_search_toolset,
    create_memory_toolset,
)
from agent_k.adapters.kaggle import KaggleAdapter
from pydantic_ai import Agent

# Supported models
SUPPORTED_MODELS = [
    'devstral:local',
    'anthropic:claude-3-haiku-20240307',
    'openrouter:mistralai/devstral-small',
]
```

### Create Toolsets

```python
# Memory for cross-agent communication
memory_toolset = create_memory_toolset(
    storage_path=Path("examples/mission_memory.json")
)

# Kaggle API access
kaggle_adapter = KaggleAdapter.from_env()
kaggle_toolset = create_kaggle_toolset(kaggle_adapter)

# Web search
search_toolset = create_search_toolset()
```

### Create LOBBYIST Agent

```python
def create_lobbyist_agent(model_spec: str) -> Agent:
    """Create the LOBBYIST discovery agent."""
    return Agent(
        get_model(model_spec),
        output_type=str,
        instructions="""You are the LOBBYIST agent.

Your mission is to discover Kaggle competitions.

TOOLS AVAILABLE:
- kaggle_search_competitions: Search Kaggle API
- web_search: Search the web
- memory_store: Save findings for other agents

WORKFLOW:
1. Search for active featured competitions
2. Score competitions by fit
3. Store the best match using memory_store("target_competition", {...})
""",
        toolsets=[kaggle_toolset, search_toolset, memory_toolset],
        retries=3,
    )
```

### Create SCIENTIST Agent

```python
def create_scientist_agent(model_spec: str) -> Agent:
    """Create the SCIENTIST research agent."""
    return Agent(
        get_model(model_spec),
        output_type=str,
        instructions="""You are the SCIENTIST agent.

Your mission is to research a Kaggle competition.

TOOLS AVAILABLE:
- memory_retrieve: Get competition from LOBBYIST
- kaggle_get_leaderboard: Analyze standings
- search_papers: Find relevant papers
- web_search: Search for tips

WORKFLOW:
1. Retrieve competition: memory_retrieve("target_competition")
2. Analyze leaderboard
3. Search for papers and winning approaches
4. Store findings: memory_store("research_findings", {...})
""",
        toolsets=[kaggle_toolset, search_toolset, memory_toolset],
        retries=3,
    )
```

### Run the Demo

```python
async def run_demo(model_spec: str = 'devstral:local'):
    """Run the multi-agent demo."""
    
    print("=" * 60)
    print("🚀 AGENT-K Multi-Agent Demo")
    print(f"📦 Model: {model_spec}")
    print("=" * 60)
    
    # Phase 1: Discovery
    print("\n🎯 Phase 1: LOBBYIST - Competition Discovery\n")
    
    lobbyist = create_lobbyist_agent(model_spec)
    discovery_result = await lobbyist.run(
        prompt="""Find active featured Kaggle competitions.
        
        Look for competitions that:
        - Are currently active
        - Are Featured or Research type
        - Have reasonable deadlines
        
        Store the best match for the SCIENTIST."""
    )
    
    print(f"Discovery Result:\n{discovery_result.data}\n")
    
    # Phase 2: Research
    print("\n🔬 Phase 2: SCIENTIST - Competition Research\n")
    
    scientist = create_scientist_agent(model_spec)
    research_result = await scientist.run(
        prompt="""Research the competition stored in memory.
        
        Analyze:
        1. Leaderboard score distribution
        2. Target score for top 10%
        3. Relevant papers and techniques
        4. Winning approaches from similar competitions
        
        Synthesize a strategy."""
    )
    
    print(f"Research Result:\n{research_result.data}\n")
    
    print("=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="devstral:local",
        choices=SUPPORTED_MODELS,
        help="Model to use"
    )
    args = parser.parse_args()
    
    asyncio.run(run_demo(args.model))
```

## Expected Output

```
============================================================
🚀 AGENT-K Multi-Agent Demo
📦 Model: anthropic:claude-3-haiku-20240307
============================================================

🎯 Phase 1: LOBBYIST - Competition Discovery

Tool calls:
  → kaggle_search_competitions(categories=["Featured"])
  → memory_store(key="target_competition", value={...})

Discovery Result:
Found 15 active featured competitions. Selected "Titanic - Machine Learning 
from Disaster" as the best match based on:
- Active deadline
- Good for learning
- Large community

🔬 Phase 2: SCIENTIST - Competition Research

Tool calls:
  → memory_retrieve(key="target_competition")
  → kaggle_get_leaderboard(competition_id="titanic")
  → search_papers(query="Titanic survival prediction")
  → memory_store(key="research_findings", value={...})

Research Result:
## Leaderboard Analysis
- Top score: 1.0000
- Median score: 0.7799
- Target (top 10%): 0.8086

## Strategy Recommendations
1. Use gradient boosting ensemble (XGBoost + LightGBM)
2. Feature engineering on family relationships
3. Handle missing values with domain-informed imputation
...

============================================================
✅ Demo Complete!
============================================================
```

## Memory File

After running, check `examples/mission_memory.json`:

```json
{
  "target_competition": {
    "id": "titanic",
    "title": "Titanic - Machine Learning from Disaster",
    "fit_score": 0.85
  },
  "research_findings": {
    "leaderboard_analysis": {
      "top_score": 1.0,
      "target_score": 0.8086
    },
    "strategy_recommendations": [...]
  }
}
```

## Customization

### Change Search Criteria

Modify the LOBBYIST prompt:

```python
discovery_result = await lobbyist.run(
    prompt="""Find Kaggle competitions that:
    - Are in the computer vision domain
    - Have at least $10,000 prize pool
    - Have at least 30 days remaining
    """
)
```

### Change Target Percentile

Modify the SCIENTIST prompt:

```python
research_result = await scientist.run(
    prompt="""Research the competition and determine:
    - What score is needed for top 5%?
    - What advanced techniques are winners using?
    """
)
```

## Troubleshooting

### Rate Limiting

If you see rate limit errors, the toolsets handle retries automatically.

### Model Errors

Try a different model:

```bash
# If local Devstral fails
uv run python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307
```

### Missing API Keys

Ensure `.env` has the required keys for your model provider.

## Next Steps

- [Custom Agent](custom-agent.md) — Create your own agent
- [Toolsets](../concepts/toolsets.md) — Learn about FunctionToolsets
- [Dashboard](../ui/dashboard.md) — Monitor missions visually

