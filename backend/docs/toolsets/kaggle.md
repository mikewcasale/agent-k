# Kaggle Toolset

The Kaggle toolset provides tools for interacting with the Kaggle API, including competition discovery, leaderboard access, and dataset management.

## Tools

| Tool | Description |
|------|-------------|
| `kaggle_search_competitions` | Search for active competitions |
| `kaggle_get_competition` | Get competition details |
| `kaggle_get_leaderboard` | Get leaderboard entries |
| `kaggle_list_datasets` | List competition datasets |

## Setup

```python
from pydantic_ai import Agent
from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.deps import KaggleDeps
from agent_k.toolsets import kaggle_toolset
from agent_k.ui.ag_ui import EventEmitter

config = KaggleSettings(
    username="your_kaggle_username",
    api_key="your_kaggle_api_key",
)

adapter = KaggleAdapter(config)

deps = KaggleDeps(
    kaggle_adapter=adapter,
    event_emitter=EventEmitter(),
)

agent = Agent(
    "anthropic:claude-3-haiku-20240307",
    deps_type=KaggleDeps,
    toolsets=[kaggle_toolset],
)

result = await agent.run(
    "List featured Kaggle competitions",
    deps=deps,
)
```

## Tools Reference

### kaggle_search_competitions

Search Kaggle for active competitions matching criteria.

```python
async def kaggle_search_competitions(
    categories: list[str] | None = None,
    keywords: list[str] | None = None,
    min_prize: int | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    ...
```

Example response:

```python
[
    {
        "id": "titanic",
        "title": "Titanic - Machine Learning from Disaster",
        "type": "getting_started",
        "metric": "accuracy",
        "days_remaining": 365,
        "prize_pool": 0,
        "tags": ["tabular"],
        "is_active": True,
    },
]
```

### kaggle_get_competition

Get detailed competition metadata.

```python
async def kaggle_get_competition(competition_id: str) -> dict[str, Any]:
    ...
```

Example response:

```python
{
    "id": "titanic",
    "title": "Titanic - Machine Learning from Disaster",
    "description": "Start here...",
    "type": "getting_started",
    "metric": "accuracy",
    "metric_direction": "maximize",
    "days_remaining": 365,
    "deadline": "2030-01-01T00:00:00Z",
    "prize_pool": 0,
    "max_team_size": 5,
    "max_daily_submissions": 5,
    "tags": ["tabular"],
}
```

### kaggle_get_leaderboard

Fetch leaderboard entries.

```python
async def kaggle_get_leaderboard(
    competition_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    ...
```

Example response:

```python
{
    "competition_id": "titanic",
    "total_entries": 20,
    "entries": [
        {"rank": 1, "team_name": "team_one", "score": 0.999},
    ],
}
```

### kaggle_list_datasets

List dataset files for the competition. This tool requires a `KaggleAdapter` because it uses the Kaggle API directly.

```python
async def kaggle_list_datasets(competition_id: str) -> dict[str, Any]:
    ...
```

Example response:

```python
{
    "competition_id": "titanic",
    "files": [
        {"name": "train.csv", "size": 59760, "description": "Training set"},
    ],
}
```

## Adapter Configuration

`KaggleSettings` exposes the core adapter configuration:

```python
from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings

config = KaggleSettings(
    username="your_username",
    api_key="your_api_key",
    base_url="https://www.kaggle.com/api/v1",
    timeout=30,
    max_retries=3,
    rate_limit_delay=1.0,
)

adapter = KaggleAdapter(config)
```

## Environment Variables

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

## API Reference

See [API Reference: KaggleToolset](../api/toolsets/kaggle.md) for complete documentation.
