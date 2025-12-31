# Search Tool Helpers

Agent-K relies on pydantic-ai's built-in `WebSearchTool` (and optional `WebFetchTool`) for web search. Custom HTTP search implementations are intentionally avoided.

## Setup

```python
from pydantic_ai import Agent

from agent_k.toolsets import prepare_web_search

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    builtin_tools=[prepare_web_search],
)
```

## Query Helpers

Use these helpers to build scoped queries for the built-in `web_search` tool:

```python
from agent_k.toolsets import build_kaggle_search_query, build_scholarly_query

kaggle_query = build_kaggle_search_query("titanic")
# -> "site:kaggle.com titanic"

papers_query = build_scholarly_query("xgboost")
# -> "site:arxiv.org OR site:paperswithcode.com xgboost"
```

## Built-in Tool Usage

The built-in tool name is `web_search`. Example prompt guidance:

```text
Call web_search(query="site:arxiv.org OR site:paperswithcode.com xgboost")
```

If you need URL fetching, add `prepare_web_fetch` to `builtin_tools` and call `web_fetch` when supported by the provider.
