"""
MCP server for Thala knowledge stores.

Entry point for the MCP server using STDIO transport.
Run with: python -m mcp.server
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from core.config import configure_langsmith

configure_langsmith()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stores.chroma import ChromaStore
from core.stores.elasticsearch import ElasticsearchStores
from core.stores.zotero import ZoteroStore

from core.embedding import EmbeddingService
from .errors import NotFoundError, StoreConnectionError, ToolError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global store instances (initialized on startup)
_stores: dict[str, Any] = {}
_embedding_service: EmbeddingService | None = None

# Create MCP server
server = Server("thala-stores")


async def init_stores():
    """Initialize all store connections."""
    global _stores, _embedding_service

    logger.info("Initializing stores...")

    # Initialize embedding service first
    try:
        _embedding_service = EmbeddingService()
        logger.info(f"Embedding service initialized: {_embedding_service.provider_name}/{_embedding_service.model}")
    except Exception as e:
        logger.warning(f"Embedding service initialization failed: {e}")
        _embedding_service = None

    # Initialize Elasticsearch stores
    es_coherence_host = os.environ.get("THALA_ES_COHERENCE_HOST", "http://localhost:9201")
    es_forgotten_host = os.environ.get("THALA_ES_FORGOTTEN_HOST", "http://localhost:9200")

    try:
        es_stores = ElasticsearchStores(
            coherence_host=es_coherence_host,
            forgotten_host=es_forgotten_host,
        )
        _stores["es"] = es_stores
        logger.info("Elasticsearch stores initialized")
    except Exception as e:
        logger.error(f"Elasticsearch initialization failed: {e}")
        _stores["es"] = None

    # Initialize ChromaDB store (requires es_stores for history tracking)
    chroma_host = os.environ.get("THALA_CHROMA_HOST", "localhost")
    chroma_port = int(os.environ.get("THALA_CHROMA_PORT", "8000"))

    try:
        chroma_store = ChromaStore(
            host=chroma_host,
            port=chroma_port,
            es_stores=_stores.get("es"),
        )
        _stores["chroma"] = chroma_store
        logger.info("ChromaDB store initialized")
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
        _stores["chroma"] = None

    # Initialize Zotero store
    zotero_host = os.environ.get("THALA_ZOTERO_HOST", "localhost")
    zotero_port = int(os.environ.get("THALA_ZOTERO_PORT", "23119"))

    try:
        zotero_store = ZoteroStore(
            host=zotero_host,
            port=zotero_port,
        )
        _stores["zotero"] = zotero_store
        logger.info("Zotero store initialized")
    except Exception as e:
        logger.error(f"Zotero initialization failed: {e}")
        _stores["zotero"] = None

    logger.info("Store initialization complete")


async def cleanup_stores():
    """Close all store connections."""
    global _stores, _embedding_service

    if _embedding_service:
        await _embedding_service.close()

    if _stores.get("es"):
        await _stores["es"].close()

    if _stores.get("zotero"):
        await _stores["zotero"].close()


def get_stores() -> dict[str, Any]:
    """Get store instances."""
    return _stores


def get_embedding_service() -> EmbeddingService | None:
    """Get embedding service instance."""
    return _embedding_service


# Import tool handlers after server is created
from .tools import coherence, forgotten, health, store, top_of_mind, who_i_was, zotero


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    tools = []

    # Health check
    tools.extend(health.get_tools())

    # Read-only stores
    tools.extend(who_i_was.get_tools())
    tools.extend(forgotten.get_tools())

    # Zotero (CRUD, no embedding)
    tools.extend(zotero.get_tools())

    # CRUD stores with embedding
    tools.extend(coherence.get_tools())
    tools.extend(store.get_tools())
    tools.extend(top_of_mind.get_tools())

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        # Route to appropriate handler
        if name.startswith("health."):
            result = await health.handle(name, arguments, _stores, _embedding_service)
        elif name.startswith("who_i_was."):
            result = await who_i_was.handle(name, arguments, _stores)
        elif name.startswith("forgotten."):
            result = await forgotten.handle(name, arguments, _stores)
        elif name.startswith("zotero."):
            result = await zotero.handle(name, arguments, _stores)
        elif name.startswith("coherence."):
            result = await coherence.handle(name, arguments, _stores, _embedding_service)
        elif name.startswith("store."):
            result = await store.handle(name, arguments, _stores, _embedding_service)
        elif name.startswith("top_of_mind."):
            result = await top_of_mind.handle(name, arguments, _stores, _embedding_service)
        else:
            raise ToolError(f"Unknown tool: {name}. Use health.check to see available tools.")

        return [TextContent(type="text", text=json.dumps(result, default=str))]

    except ToolError as e:
        # Return execution error with actionable message
        return [TextContent(
            type="text",
            text=json.dumps({"error": e.message, "details": e.details}),
        )]
    except Exception as e:
        logger.exception(f"Unexpected error in tool {name}")
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Internal error: {str(e)}"}),
        )]


async def main():
    """Run the MCP server."""
    await init_stores()

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await cleanup_stores()


if __name__ == "__main__":
    asyncio.run(main())
