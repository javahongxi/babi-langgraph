"""Web search tool using Tavily Search API.

Provides real-time web search results optimized for AI agents.
Requires TAVILY_API_KEY environment variable (free tier: 1000 calls/month).
Get your API key from: https://tavily.com
"""

from __future__ import annotations

import os

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5, days: int = 0) -> str:
    """Search the web for real-time information using Tavily AI search.

    Use this tool when you need up-to-date information about current events,
    news, weather, prices, or any topic that requires recent data beyond
    your training cutoff.

    Args:
        query: The search query string. Be specific and concise for best results.
            For time-sensitive queries, include the year (e.g., "2026年最新电影").
        max_results: Maximum number of results to return (default 5, max 10).
        days: Limit results to the last N days (0 = no limit, use 7 for recent news,
            30 for monthly updates, etc.). Helps get more current results.
    """
    max_results = min(max(1, max_results), 10)

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "Error: TAVILY_API_KEY environment variable not set.\n"
            "Get your free API key from: https://tavily.com\n"
            "Then set it with: export TAVILY_API_KEY=your_api_key"
        )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        # Build search params with optional time filter
        search_params = {"query": query, "max_results": max_results}
        if days > 0:
            search_params["days"] = days
        response = client.search(**search_params)

        results = response.get("results", [])
        if not results:
            return "No search results found. Try rephrasing your query."

        output_parts = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            content = r.get("content", "")
            url = r.get("url", "")
            output_parts.append(f"[{i}] {title}\n    {content}\n    URL: {url}")

        return "\n\n".join(output_parts)

    except ImportError:
        return "Error: tavily-python package not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"Search failed: {e}"
