# Search Toolset

The Search toolset provides web and academic search capabilities, allowing agents to discover information from the web, academic papers, and Kaggle discussions.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo |
| `search_papers` | Search academic papers |
| `search_kaggle` | Search Kaggle discussions and notebooks |

## Setup

```python
from agent_k.toolsets import create_search_toolset

search_toolset = create_search_toolset()

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    toolsets=[search_toolset],
)
```

## Tools Reference

### web_search

Search the web using DuckDuckGo.

```python
@toolset.tool
async def web_search(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search the web.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
    
    Returns:
        List of search results with title, snippet, url
    """
```

**Example Response:**

```python
[
    {
        "title": "Titanic Kaggle Competition Guide",
        "snippet": "A comprehensive guide to the Titanic competition...",
        "url": "https://example.com/titanic-guide",
    },
    # ...
]
```

### search_papers

Search academic papers using Semantic Scholar or similar APIs.

```python
@toolset.tool
async def search_papers(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search academic papers.
    
    Args:
        query: Search query (keywords, title, author)
        max_results: Maximum papers to return
    
    Returns:
        List of papers with title, authors, abstract, url
    """
```

**Example Response:**

```python
[
    {
        "title": "XGBoost: A Scalable Tree Boosting System",
        "authors": ["Tianqi Chen", "Carlos Guestrin"],
        "abstract": "Tree boosting is a highly effective and widely used...",
        "year": 2016,
        "url": "https://arxiv.org/abs/1603.02754",
        "citations": 15000,
    },
    # ...
]
```

### search_kaggle

Search Kaggle discussions, notebooks, and datasets.

```python
@toolset.tool
async def search_kaggle(
    query: str,
    search_type: str = "discussions",
) -> list[dict[str, Any]]:
    """Search Kaggle content.
    
    Args:
        query: Search query
        search_type: One of "discussions", "notebooks", "datasets"
    
    Returns:
        List of results matching query
    """
```

**Example Response (discussions):**

```python
[
    {
        "title": "How I got 0.85+ accuracy",
        "author": "grandmaster_user",
        "url": "https://kaggle.com/competitions/titanic/discussion/12345",
        "votes": 500,
        "comments": 120,
        "content_preview": "Here's my approach to the Titanic competition...",
    },
    # ...
]
```

**Example Response (notebooks):**

```python
[
    {
        "title": "Titanic EDA + Feature Engineering",
        "author": "data_scientist",
        "url": "https://kaggle.com/code/data_scientist/titanic-eda",
        "votes": 1000,
        "execution_count": 5000,
        "language": "Python",
    },
    # ...
]
```

## Implementation Details

### DuckDuckGo Search

Uses the `duckduckgo-search` library:

```python
from duckduckgo_search import AsyncDDGS

async def web_search(query: str, max_results: int = 10):
    async with AsyncDDGS() as ddgs:
        results = []
        async for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r["title"],
                "snippet": r["body"],
                "url": r["href"],
            })
        return results
```

### Academic Search

Uses scholarly or Semantic Scholar API:

```python
import httpx

async def search_papers(query: str, max_results: int = 10):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": max_results},
        )
        data = resp.json()
        return [
            {
                "title": p["title"],
                "authors": [a["name"] for a in p.get("authors", [])],
                "abstract": p.get("abstract", ""),
                "year": p.get("year"),
                "url": p.get("url"),
            }
            for p in data.get("data", [])
        ]
```

## Caching

Search results can be cached to reduce API calls:

```python
from functools import lru_cache
from datetime import datetime, timedelta

_cache = {}

async def web_search(query: str, max_results: int = 10):
    cache_key = f"web:{query}:{max_results}"
    
    if cache_key in _cache:
        cached, timestamp = _cache[cache_key]
        if datetime.now() - timestamp < timedelta(hours=1):
            return cached
    
    results = await _do_search(query, max_results)
    _cache[cache_key] = (results, datetime.now())
    
    return results
```

## Rate Limiting

Respect rate limits on external APIs:

```python
from asyncio import Semaphore

_semaphore = Semaphore(5)  # Max 5 concurrent searches

async def web_search(...):
    async with _semaphore:
        return await _do_search(...)
```

## Error Handling

Handle common search errors:

```python
@toolset.tool
async def web_search(...):
    try:
        return await _do_search(query, max_results)
    except RateLimitError:
        return {"error": "Search rate limited. Try again later."}
    except TimeoutError:
        return {"error": "Search timed out."}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
```

## Testing

```python
@pytest.fixture
def mock_search():
    with patch("agent_k.toolsets.search._do_search") as mock:
        mock.return_value = [
            {"title": "Result 1", "snippet": "...", "url": "..."}
        ]
        yield mock

async def test_web_search(mock_search):
    toolset = create_search_toolset()
    results = await toolset.get_tool("web_search")(query="test")
    assert len(results) == 1
```

## API Reference

See [API Reference: SearchToolset](../api/toolsets/search.md) for complete documentation.

