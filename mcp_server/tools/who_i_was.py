"""who_i_was tools - READ ONLY access to edit history."""

from typing import Any

from mcp.types import Tool

from ..errors import NotFoundError, ToolError
from ..response_utils import format_search_results, format_single_record
from ..store_utils import get_es_substore
from ..validation_utils import parse_uuid_arg


def get_tools() -> list[Tool]:
    """Get who_i_was tools (read-only)."""
    return [
        Tool(
            name="who_i_was.get",
            description="Retrieve a historical snapshot by UUID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the historical snapshot",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="who_i_was.get_history",
            description="Get all historical versions that superseded a specific record. Returns edit history for a record.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the original record to get history for",
                    },
                },
                "required": ["record_id"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="who_i_was.search",
            description="Search historical records using Elasticsearch query DSL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": "Elasticsearch query DSL object (e.g., {'match': {'reason': 'updated'}})",
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
    """Handle who_i_was tool calls."""
    who_i_was_store = get_es_substore(stores, "who_i_was")

    if name == "who_i_was.get":
        record_id = parse_uuid_arg(arguments)
        record = await who_i_was_store.get(record_id)
        if record is None:
            raise NotFoundError("who_i_was", arguments["id"])
        return format_single_record(record)

    elif name == "who_i_was.get_history":
        record_id = parse_uuid_arg(arguments, "record_id")
        records = await who_i_was_store.get_history(record_id)
        return {
            "record_id": arguments["record_id"],
            "history_count": len(records),
            "history": [r.model_dump(mode="json") for r in records],
        }

    elif name == "who_i_was.search":
        query = arguments["query"]
        size = arguments.get("size", 10)
        records = await who_i_was_store.search(query, size=size)
        return format_search_results(records)

    else:
        raise ToolError(f"Unknown who_i_was tool: {name}")
