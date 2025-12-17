"""Health check tool for MCP server."""

from typing import Any

from mcp.types import Tool

from ..errors import ToolError


def get_tools() -> list[Tool]:
    """Get health check tools."""
    return [
        Tool(
            name="health.check",
            description="Check connectivity to all backend services (Elasticsearch, ChromaDB, Zotero, Embedding).",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    ]


async def handle(
    name: str,
    arguments: dict[str, Any],
    stores: dict[str, Any],
    embedding_service: Any | None,
) -> dict[str, Any]:
    """Handle health check tool calls."""
    if name != "health.check":
        raise ToolError(f"Unknown health tool: {name}")

    results: dict[str, Any] = {}

    # Check Elasticsearch
    es_stores = stores.get("es")
    if es_stores:
        try:
            es_health = await es_stores.health_check()
            results["elasticsearch"] = es_health
        except Exception as e:
            results["elasticsearch"] = {"healthy": False, "error": str(e)}
    else:
        results["elasticsearch"] = {"healthy": False, "error": "Not initialized"}

    # Check ChromaDB
    chroma_store = stores.get("chroma")
    if chroma_store:
        try:
            chroma_healthy = await chroma_store.health_check()
            results["chroma"] = {"healthy": chroma_healthy}
        except Exception as e:
            results["chroma"] = {"healthy": False, "error": str(e)}
    else:
        results["chroma"] = {"healthy": False, "error": "Not initialized"}

    # Check Zotero
    zotero_store = stores.get("zotero")
    if zotero_store:
        try:
            zotero_health = await zotero_store.health_check()
            results["zotero"] = {
                "healthy": zotero_health.healthy,
                "plugin": zotero_health.plugin,
                "version": zotero_health.version,
            }
            if zotero_health.error:
                results["zotero"]["error"] = zotero_health.error
        except Exception as e:
            results["zotero"] = {"healthy": False, "error": str(e)}
    else:
        results["zotero"] = {"healthy": False, "error": "Not initialized"}

    # Check Embedding service
    if embedding_service:
        results["embedding"] = {
            "healthy": True,
            "provider": embedding_service.provider_name,
            "model": embedding_service.model,
        }
    else:
        results["embedding"] = {"healthy": False, "error": "Not initialized"}

    # Overall health
    results["overall_healthy"] = all(
        r.get("healthy", False) for r in results.values() if isinstance(r, dict)
    )

    return results
