"""MCP server wrapping paper corpus tools for CLI-based invocation.

Exposes search_papers and get_paper_content (and optionally check_fact)
as MCP tools so that `claude -p --mcp-config` can use them during
agentic tool-calling loops. This avoids direct API billing for tool-
calling enhance/fact-check nodes.

Run with: python -m mcp_server.paper_tools
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

# Tools to expose, controlled by env var (comma-separated).
# Default: search_papers,get_paper_content
_ENABLED_TOOLS = set(
    os.getenv("THALA_MCP_TOOLS", "search_papers,get_paper_content").split(",")
)

server = Server("paper-tools")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOL_DEFS: dict[str, Tool] = {
    "search_papers": Tool(
        name="search_papers",
        description=(
            "Search available papers by topic using hybrid search. "
            "Combines semantic and keyword search. Returns brief metadata "
            "for papers matching the query. Use get_paper_content to fetch "
            "detailed content for specific papers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic, keyword, or concept to search for",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum papers to return (default 10)",
                },
            },
            "required": ["query"],
        },
    ),
    "get_paper_content": Tool(
        name="get_paper_content",
        description=(
            "Fetch detailed content for a specific paper. Returns the "
            "10:1 compressed summary which captures key content. "
            "Use after search_papers identifies relevant papers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "zotero_key": {
                    "type": "string",
                    "description": "Paper citation key from search_papers results (8 alphanumeric chars)",
                },
                "max_chars": {
                    "type": "integer",
                    "default": 10000,
                    "minimum": 1000,
                    "maximum": 20000,
                    "description": "Maximum content length (default 10000)",
                },
            },
            "required": ["zotero_key"],
        },
    ),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List enabled tools."""
    return [t for name, t in _TOOL_DEFS.items() if name in _ENABLED_TOOLS]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool call by delegating to the LangChain tool implementations."""
    if name not in _ENABLED_TOOLS:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    try:
        if name == "search_papers":
            from langchain_tools.paper_corpus import search_papers

            result = await search_papers.ainvoke(
                {"query": arguments["query"], "limit": arguments.get("limit", 10)}
            )
        elif name == "get_paper_content":
            from langchain_tools.paper_corpus import get_paper_content

            result = await get_paper_content.ainvoke(
                {
                    "zotero_key": arguments["zotero_key"],
                    "max_chars": arguments.get("max_chars", 10000),
                }
            )
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unhandled tool: {name}"}))]

        # LangChain tools return dicts via output_dict()
        text = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
        return [TextContent(type="text", text=text)]

    except Exception as e:
        logger.exception(f"Tool {name} failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    """Run the paper tools MCP server."""
    logger.info(f"Paper tools MCP server starting (tools: {_ENABLED_TOOLS})")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
