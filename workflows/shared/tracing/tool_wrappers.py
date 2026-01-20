"""Traced wrapper functions for tool calls.

These wrappers provide LangSmith visibility into tool execution for evaluation purposes.
Each wrapper is decorated with @traceable(run_type="tool") to appear as tool spans in traces.
"""

import logging
from typing import Any

from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="execute_tool_call")
async def traced_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_func: Any,
) -> Any:
    """Generic traced wrapper for any tool call.

    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments passed to the tool
        tool_func: The tool function or coroutine to execute

    Returns:
        Tool result
    """
    logger.debug(f"Executing tool: {tool_name} with args: {list(tool_args.keys())}")
    if hasattr(tool_func, "ainvoke"):
        result = await tool_func.ainvoke(tool_args)
    elif callable(tool_func):
        import asyncio

        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**tool_args)
        else:
            result = tool_func(**tool_args)
    else:
        raise ValueError(f"Tool {tool_name} is not callable")

    return result


@traceable(run_type="tool", name="search_papers", tags=["tool:search"])
async def traced_search_papers(
    query: str,
    max_results: int = 10,
    **kwargs: Any,
) -> Any:
    """Traced wrapper for paper search operations.

    Args:
        query: Search query string
        max_results: Maximum results to return
        **kwargs: Additional search parameters

    Returns:
        Search results from the paper search tool
    """
    from langchain_tools import search_papers

    return await search_papers.ainvoke({
        "query": query,
        "max_results": max_results,
        **kwargs,
    })


@traceable(run_type="tool", name="get_paper_content", tags=["tool:retrieval"])
async def traced_get_paper_content(
    zotero_key: str,
    **kwargs: Any,
) -> Any:
    """Traced wrapper for retrieving paper content.

    Args:
        zotero_key: Zotero key of the paper to retrieve
        **kwargs: Additional retrieval parameters

    Returns:
        Paper content from the retrieval tool
    """
    from langchain_tools import get_paper_content

    return await get_paper_content.ainvoke({
        "zotero_key": zotero_key,
        **kwargs,
    })


@traceable(run_type="tool", name="web_search", tags=["tool:web"])
async def traced_web_search(
    query: str,
    max_results: int = 10,
    **kwargs: Any,
) -> Any:
    """Traced wrapper for web search operations.

    Args:
        query: Search query string
        max_results: Maximum results to return
        **kwargs: Additional search parameters

    Returns:
        Search results from the web search tool
    """
    from langchain_tools import web_search

    return await web_search.ainvoke({
        "query": query,
        "max_results": max_results,
        **kwargs,
    })


@traceable(run_type="tool", name="scrape_url", tags=["tool:web"])
async def traced_scrape_url(
    url: str,
    **kwargs: Any,
) -> Any:
    """Traced wrapper for URL scraping operations.

    Args:
        url: URL to scrape
        **kwargs: Additional scraping parameters

    Returns:
        Scraped content from the URL
    """
    from langchain_tools import scrape_url

    return await scrape_url.ainvoke({
        "url": url,
        **kwargs,
    })
