"""Web search toolset for AGENT-K agents.

Provides web search functionality as a pydantic-ai toolset that works
with any model provider.

Uses FunctionToolset to properly integrate with pydantic-ai's tool system.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

import httpx
import logfire
from pydantic_ai.toolsets import FunctionToolset

__all__ = ['create_search_toolset']


def create_search_toolset(
    http_client: httpx.AsyncClient | None = None,
) -> FunctionToolset[Any]:
    """Create a web search toolset.
    
    This creates a FunctionToolset with tools for web search.
    Uses DuckDuckGo (no API key required).
    Works with any model provider including OpenAI-compatible endpoints.
    
    Example:
        >>> toolset = create_search_toolset()
        >>> agent = Agent('devstral:local', toolsets=[toolset])
    """
    toolset: FunctionToolset[Any] = FunctionToolset(id='web_search')
    
    # Cache for search results
    _cache: dict[str, list[dict[str, str]]] = {}
    _client: httpx.AsyncClient | None = http_client
    
    async def _get_client() -> httpx.AsyncClient:
        nonlocal _client
        if _client is None:
            _client = httpx.AsyncClient(timeout=30, follow_redirects=True)
        return _client
    
    async def _duckduckgo_search(
        query: str,
        num_results: int,
    ) -> list[dict[str, str]]:
        """Search using DuckDuckGo HTML interface."""
        try:
            client = await _get_client()
            url = 'https://html.duckduckgo.com/html/'
            data = {'q': query, 's': '0'}
            
            response = await client.post(
                url,
                data=data,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/120.0.0.0 Safari/537.36',
                },
            )
            
            if response.status_code != 200:
                logfire.warning('duckduckgo_search_failed', status=response.status_code)
                return []
            
            return _parse_duckduckgo_html(response.text)[:num_results]
            
        except Exception as e:
            logfire.error('web_search_error', error=str(e))
            return []
    
    def _parse_duckduckgo_html(html: str) -> list[dict[str, str]]:
        """Parse search results from DuckDuckGo HTML."""
        results = []
        
        # Find result links
        links = re.findall(r'href="//duckduckgo\.com/l/\?uddg=([^&"]+)', html)
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
        
        for i, (link, title) in enumerate(zip(links, titles)):
            url = unquote(link)
            snippet = snippets[i] if i < len(snippets) else ''
            
            results.append({
                'title': title.strip(),
                'url': url,
                'snippet': snippet.strip(),
            })
        
        return results
    
    @toolset.tool
    async def web_search(
        query: str,
        num_results: int = 5,
    ) -> dict[str, Any]:
        """Search the web for information.
        
        Args:
            query: The search query
            num_results: Maximum number of results (default: 5)
        
        Returns:
            Search results with titles, URLs, and snippets.
        """
        with logfire.span('web_search', query=query):
            # Check cache
            cache_key = f'web:{query.lower().strip()}'
            if cache_key in _cache:
                return {
                    'query': query,
                    'cached': True,
                    'results': _cache[cache_key][:num_results],
                }
            
            results = await _duckduckgo_search(query, num_results)
            _cache[cache_key] = results
            
            return {
                'query': query,
                'cached': False,
                'count': len(results),
                'results': results,
            }
    
    @toolset.tool
    async def search_kaggle(
        query: str,
    ) -> dict[str, Any]:
        """Search specifically for Kaggle content.
        
        Args:
            query: Search query (will be scoped to kaggle.com)
        
        Returns:
            Kaggle-specific search results.
        """
        with logfire.span('search_kaggle', query=query):
            full_query = f'site:kaggle.com {query}'
            
            cache_key = f'kaggle:{query.lower().strip()}'
            if cache_key in _cache:
                return {
                    'query': query,
                    'cached': True,
                    'results': _cache[cache_key],
                }
            
            results = await _duckduckgo_search(full_query, 10)
            _cache[cache_key] = results
            
            return {
                'query': query,
                'cached': False,
                'count': len(results),
                'results': results,
            }
    
    @toolset.tool
    async def search_papers(
        topic: str,
        source: str = 'all',
    ) -> dict[str, Any]:
        """Search for academic papers on arXiv or Papers with Code.
        
        Args:
            topic: Research topic to search for
            source: Source to search - 'arxiv', 'paperswithcode', or 'all'
        
        Returns:
            Academic paper search results.
        """
        with logfire.span('search_papers', topic=topic, source=source):
            if source == 'arxiv':
                query = f'site:arxiv.org {topic}'
            elif source == 'paperswithcode':
                query = f'site:paperswithcode.com {topic}'
            else:
                query = f'site:arxiv.org OR site:paperswithcode.com {topic}'
            
            cache_key = f'papers:{source}:{topic.lower().strip()}'
            if cache_key in _cache:
                return {
                    'topic': topic,
                    'source': source,
                    'cached': True,
                    'results': _cache[cache_key],
                }
            
            results = await _duckduckgo_search(query, 10)
            _cache[cache_key] = results
            
            return {
                'topic': topic,
                'source': source,
                'cached': False,
                'count': len(results),
                'results': results,
            }
    
    return toolset
