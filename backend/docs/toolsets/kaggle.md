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
from agent_k.adapters.kaggle import KaggleAdapter, KaggleConfig
from agent_k.toolsets import create_kaggle_toolset

# Configure adapter
config = KaggleConfig(
    username="your_kaggle_username",
    api_key="your_kaggle_api_key",
)

# Create adapter and toolset
adapter = KaggleAdapter(config)
kaggle_toolset = create_kaggle_toolset(adapter)

# Use with agent
agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    toolsets=[kaggle_toolset],
)
```

## Tools Reference

### kaggle_search_competitions

Search Kaggle for active competitions matching criteria.

```python
@toolset.tool
async def kaggle_search_competitions(
    categories: list[str] | None = None,
    keywords: list[str] | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """Search Kaggle for active competitions.
    
    Args:
        categories: Filter by competition type (Featured, Research, etc.)
        keywords: Keywords to search in title/description
        active_only: Only return currently active competitions
    
    Returns:
        List of competition dictionaries with id, title, deadline, etc.
    """
```

**Example Response:**

```python
[
    {
        "id": "titanic",
        "title": "Titanic - Machine Learning from Disaster",
        "category": "Getting Started",
        "reward": "$0",
        "deadline": "2030-01-01T00:00:00Z",
        "team_count": 50000,
        "kernel_count": 10000,
    },
    # ...
]
```

### kaggle_get_competition

Get detailed information about a specific competition.

```python
@toolset.tool
async def kaggle_get_competition(
    competition_id: str,
) -> dict[str, Any]:
    """Get competition details.
    
    Args:
        competition_id: The competition slug (e.g., "titanic")
    
    Returns:
        Competition details including description, rules, evaluation metric
    """
```

**Example Response:**

```python
{
    "id": "titanic",
    "title": "Titanic - Machine Learning from Disaster",
    "description": "Start here! Predict survival on the Titanic...",
    "evaluation_metric": "accuracy",
    "deadline": "2030-01-01T00:00:00Z",
    "max_team_size": 5,
    "rules": "...",
}
```

### kaggle_get_leaderboard

Get leaderboard entries for a competition.

```python
@toolset.tool
async def kaggle_get_leaderboard(
    competition_id: str,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Get competition leaderboard.
    
    Args:
        competition_id: The competition slug
        page_size: Number of entries to return
    
    Returns:
        List of leaderboard entries with rank, team, score
    """
```

**Example Response:**

```python
[
    {"rank": 1, "team_name": "grandmaster_1", "score": 0.99999},
    {"rank": 2, "team_name": "data_wizard", "score": 0.99998},
    # ...
]
```

### kaggle_list_datasets

List available datasets for a competition.

```python
@toolset.tool
async def kaggle_list_datasets(
    competition_id: str,
) -> list[dict[str, Any]]:
    """List competition datasets.
    
    Args:
        competition_id: The competition slug
    
    Returns:
        List of dataset files with name, size, description
    """
```

**Example Response:**

```python
[
    {"name": "train.csv", "size": 59760, "description": "Training set"},
    {"name": "test.csv", "size": 27960, "description": "Test set"},
    {"name": "sample_submission.csv", "size": 2780, "description": "Sample submission"},
]
```

## Adapter Configuration

The `KaggleAdapter` handles API authentication and rate limiting:

```python
from agent_k.adapters.kaggle import KaggleAdapter, KaggleConfig

config = KaggleConfig(
    username="your_username",     # KAGGLE_USERNAME env var
    api_key="your_api_key",       # KAGGLE_KEY env var
    rate_limit_per_minute=100,    # Default: 100
    timeout_seconds=30,           # Default: 30
)

adapter = KaggleAdapter(config)
```

## Environment Variables

```bash
# Required
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Or use `~/.kaggle/kaggle.json`:

```json
{
    "username": "your_username",
    "key": "your_api_key"
}
```

## Error Handling

The toolset handles common errors:

```python
@toolset.tool
async def kaggle_search_competitions(...):
    try:
        results = await adapter.search_competitions(...)
        return [r.model_dump() for r in results]
    except RateLimitError:
        return {"error": "Rate limited. Try again in 60 seconds."}
    except AuthenticationError:
        return {"error": "Invalid Kaggle credentials."}
    except CompetitionNotFoundError:
        return {"error": f"Competition not found: {competition_id}"}
```

## Testing

Mock the adapter for testing:

```python
from unittest.mock import AsyncMock

@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.search_competitions.return_value = [
        Competition(id="titanic", title="Titanic", ...)
    ]
    return adapter

async def test_kaggle_search(mock_adapter):
    toolset = create_kaggle_toolset(mock_adapter)
    results = await toolset.get_tool("kaggle_search_competitions")(
        categories=["Featured"]
    )
    assert len(results) == 1
```

## API Reference

See [API Reference: KaggleToolset](../api/toolsets/kaggle.md) for complete documentation.

