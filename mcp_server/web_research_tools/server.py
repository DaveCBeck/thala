"""MCP server wrapping web research tools for CLI-based invocation.

Exposes web_search (Firecrawl), perplexity_search, and scrape_url
as MCP tools so `claude -p --mcp-config` can use them for agentic
web research.

Run with: python -m mcp_server.web_research_tools
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from core.config import configure_logging

configure_logging()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

_ENABLED_TOOLS = set(
    os.getenv("THALA_MCP_TOOLS", "web_search,perplexity_search,scrape_url").split(",")
)

server = Server("web-research-tools")

_TOOL_DEFS: dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        description=(
            "Search the web via Firecrawl. Returns titles, URLs, and snippets. "
            "Use specific, domain-appropriate search terms. "
            "Good for news, company pages, industry reports, trade press. "
            "Prefer authoritative sources over SEO content farms."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — use domain-specific terminology",
                },
            },
            "required": ["query"],
        },
    ),
    "perplexity_search": Tool(
        name="perplexity_search",
        description=(
            "AI-powered web search via Perplexity. Returns synthesized answers "
            "with source URLs. Good for getting overviews and discovering key sources. "
            "Use natural-language questions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or natural-language question",
                },
            },
            "required": ["query"],
        },
    ),
    "scrape_url": Tool(
        name="scrape_url",
        description=(
            "Fetch full text content of a web page. Use on promising URLs from "
            "search results to get detailed information. Returns page text "
            "(truncated to ~15000 chars if very long). Will reject non-webpage "
            "URLs (data files, archives)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from",
                },
            },
            "required": ["url"],
        },
    ),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [t for name, t in _TOOL_DEFS.items() if name in _ENABLED_TOOLS]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name not in _ENABLED_TOOLS:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    try:
        if name == "web_search":
            result = await _web_search(arguments["query"])
        elif name == "perplexity_search":
            result = await _perplexity_search(arguments["query"])
        elif name == "scrape_url":
            result = await _scrape_url(arguments["url"])
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unhandled: {name}"}))]

        return [TextContent(type="text", text=result)]

    except Exception as e:
        logger.exception(f"Tool {name} failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _web_search(query: str) -> str:
    from langchain_tools.firecrawl import web_search

    result = await web_search.ainvoke({"query": query, "limit": 5})
    entries = result.get("results", [])
    if not entries:
        return "No results found."
    lines = []
    for r in entries:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        desc = r.get("description", "No description")
        lines.append(f"- [{title}]({url}): {desc}")
    return "\n".join(lines)


async def _perplexity_search(query: str) -> str:
    from langchain_tools.perplexity import perplexity_search

    result = await perplexity_search.ainvoke({"query": query, "limit": 5})
    entries = result.get("results", [])
    if not entries:
        return "No results found."
    lines = []
    for r in entries:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", r.get("description", ""))
        lines.append(f"- [{title}]({url}): {snippet}")
    return "\n".join(lines)


async def _scrape_url(url: str) -> str:
    from core.scraping import get_url, GetUrlOptions
    from workflows.research.web_research.subgraphs.researcher_base.url_scraper import (
        is_scrapable_url,
    )

    if not is_scrapable_url(url):
        return f"Rejected: {url} appears to be a data file, not a web page."
    try:
        result = await get_url(
            url, GetUrlOptions(detect_academic=False, allow_retrieve_academic=False)
        )
        content = result.content
        if len(content) > 15000:
            content = content[:15000] + "\n\n[Content truncated at 15000 chars]"
        return content
    except Exception as e:
        return f"Failed to scrape {url}: {e}"


async def main():
    logger.info(f"Web research tools MCP server starting (tools: {_ENABLED_TOOLS})")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
