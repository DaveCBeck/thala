"""coherence tools - Full CRUD for beliefs, preferences, identity with embedding support."""

from typing import Any
from uuid import UUID

from mcp.types import Tool

from ..errors import EmbeddingError, NotFoundError, StoreConnectionError, ToolError, ValidationError


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

    es_stores = stores.get("es")
    if not es_stores:
        raise StoreConnectionError("elasticsearch", "Elasticsearch stores not initialized")

    coherence_store = es_stores.coherence

    if name == "coherence.add":
        # Generate embedding if service available
        embedding_model = None
        if embedding_service:
            try:
                await embedding_service.embed(arguments["content"])  # Validate embedding works
                embedding_model = embedding_service.model
            except Exception as e:
                raise EmbeddingError(str(e), embedding_service.provider_name if embedding_service else None)

        # Determine source type based on zotero_key
        source_type = SourceType.EXTERNAL if arguments.get("zotero_key") else SourceType.INTERNAL

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
        record_id = UUID(arguments["id"])
        record = await coherence_store.get(record_id)
        if record is None:
            raise NotFoundError("coherence", arguments["id"])
        return record.model_dump(mode="json")

    elif name == "coherence.update":
        record_id = UUID(arguments["id"])

        # Build updates dict
        updates: dict[str, Any] = {"_change_reason": arguments["reason"]}

        if "content" in arguments:
            updates["content"] = arguments["content"]
            # Regenerate embedding if content changes
            if embedding_service:
                try:
                    await embedding_service.embed(arguments["content"])
                    updates["embedding_model"] = embedding_service.model
                except Exception as e:
                    raise EmbeddingError(str(e), embedding_service.provider_name if embedding_service else None)

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
        record_id = UUID(arguments["id"])
        reason = arguments["reason"]

        success = await coherence_store.delete(record_id, reason=reason)
        if not success:
            raise NotFoundError("coherence", arguments["id"])

        return {"id": arguments["id"], "deleted": True}

    elif name == "coherence.search":
        query = arguments["query"]
        size = arguments.get("size", 10)
        records = await coherence_store.search(query, size=size)
        return {
            "count": len(records),
            "results": [r.model_dump(mode="json") for r in records],
        }

    else:
        raise ToolError(f"Unknown coherence tool: {name}")
