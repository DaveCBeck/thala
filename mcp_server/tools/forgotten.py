"""forgotten tools - READ ONLY access to archived content."""

from typing import Any
from uuid import UUID

from mcp.types import Tool

from ..errors import NotFoundError, StoreConnectionError, ToolError


def get_tools() -> list[Tool]:
    """Get forgotten store tools (read-only)."""
    return [
        Tool(
            name="forgotten.get",
            description="Retrieve an archived/forgotten record by UUID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the forgotten record",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="forgotten.search",
            description="Search forgotten/archived records using Elasticsearch query DSL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": "Elasticsearch query DSL object (e.g., {'match': {'forgotten_reason': 'outdated'}})",
                    },
                    "size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum number of results to return",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
    ]


async def handle(
    name: str,
    arguments: dict[str, Any],
    stores: dict[str, Any],
) -> dict[str, Any]:
    """Handle forgotten tool calls."""
    es_stores = stores.get("es")
    if not es_stores:
        raise StoreConnectionError("elasticsearch", "Elasticsearch stores not initialized")

    forgotten_store = es_stores.forgotten

    if name == "forgotten.get":
        record_id = UUID(arguments["id"])
        record = await forgotten_store.get(record_id)
        if record is None:
            raise NotFoundError("forgotten", arguments["id"])
        return record.model_dump(mode="json")

    elif name == "forgotten.search":
        query = arguments["query"]
        size = arguments.get("size", 10)
        records = await forgotten_store.search(query, size=size)
        return {
            "count": len(records),
            "results": [r.model_dump(mode="json") for r in records],
        }

    else:
        raise ToolError(f"Unknown forgotten tool: {name}")
