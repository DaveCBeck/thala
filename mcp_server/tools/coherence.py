"""coherence tools - Full CRUD for beliefs, preferences, identity with embedding support."""

from typing import Any

from mcp.types import Tool

from ..embedding_utils import generate_embedding
from ..errors import NotFoundError, ToolError
from ..response_utils import format_search_results, format_single_record
from ..store_utils import get_es_substore
from ..validation_utils import parse_uuid_arg


def get_tools() -> list[Tool]:
    """Get coherence tools (full CRUD with embedding)."""
    return [
        Tool(
            name="coherence.add",
            description="Add a new coherence record (beliefs, preferences, identity). Automatically generates embedding from content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The belief, preference, or identity statement text",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score (0.0 to 1.0)",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["belief", "preference", "identity", "goal"],
                        "description": "Category of the coherence record",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata (source, related topics, etc.)",
                    },
                    "zotero_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8,
                        "description": "Optional linked Zotero item (8-char key)",
                    },
                },
                "required": ["content", "confidence"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="coherence.get",
            description="Retrieve a coherence record by UUID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the coherence record",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="coherence.update",
            description="Update a coherence record. Automatically creates snapshot in who_i_was before updating. Regenerates embedding if content changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the coherence record to update",
                    },
                    "content": {
                        "type": "string",
                        "description": "Updated content (will regenerate embedding)",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Updated confidence score",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["belief", "preference", "identity", "goal"],
                        "description": "Updated category",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Updated metadata",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the update (stored in who_i_was)",
                    },
                },
                "required": ["id", "reason"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="coherence.delete",
            description="Delete a coherence record. Archives snapshot to who_i_was before deletion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the coherence record to delete",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for deletion (required, stored in who_i_was)",
                    },
                },
                "required": ["id", "reason"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="coherence.search",
            description="Search coherence records using Elasticsearch query DSL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": "Elasticsearch query DSL object (e.g., {'match': {'content': 'learning'}})",
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
    embedding_service: Any | None,
) -> dict[str, Any]:
    """Handle coherence tool calls."""
    from core.stores.schema import CoherenceRecord, SourceType

    coherence_store = get_es_substore(stores, "coherence")

    if name == "coherence.add":
        _, embedding_model = await generate_embedding(
            embedding_service, arguments["content"]
        )

        source_type = (
            SourceType.EXTERNAL if arguments.get("zotero_key") else SourceType.INTERNAL
        )

        record = CoherenceRecord(
            source_type=source_type,
            content=arguments["content"],
            confidence=arguments["confidence"],
            category=arguments.get("category"),
            metadata=arguments.get("metadata", {}),
            zotero_key=arguments.get("zotero_key"),
            embedding_model=embedding_model,
        )

        record_id = await coherence_store.add(record)
        return {"id": str(record_id), "success": True}

    elif name == "coherence.get":
        record_id = parse_uuid_arg(arguments)
        record = await coherence_store.get(record_id)
        if record is None:
            raise NotFoundError("coherence", arguments["id"])
        return format_single_record(record)

    elif name == "coherence.update":
        record_id = parse_uuid_arg(arguments)

        updates: dict[str, Any] = {"_change_reason": arguments["reason"]}

        if "content" in arguments:
            updates["content"] = arguments["content"]
            _, embedding_model = await generate_embedding(
                embedding_service, arguments["content"]
            )
            if embedding_model:
                updates["embedding_model"] = embedding_model

        if "confidence" in arguments:
            updates["confidence"] = arguments["confidence"]

        if "category" in arguments:
            updates["category"] = arguments["category"]

        if "metadata" in arguments:
            updates["metadata"] = arguments["metadata"]

        success = await coherence_store.update(record_id, updates)
        if not success:
            raise NotFoundError("coherence", arguments["id"])

        return {"id": arguments["id"], "success": True}

    elif name == "coherence.delete":
        record_id = parse_uuid_arg(arguments)
        reason = arguments["reason"]

        success = await coherence_store.delete(record_id, reason=reason)
        if not success:
            raise NotFoundError("coherence", arguments["id"])

        return {"id": arguments["id"], "deleted": True}

    elif name == "coherence.search":
        query = arguments["query"]
        size = arguments.get("size", 10)
        records = await coherence_store.search(query, size=size)
        return format_search_results(records)

    else:
        raise ToolError(f"Unknown coherence tool: {name}")
