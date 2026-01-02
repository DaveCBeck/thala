"""store tools - Full CRUD for main knowledge store with embedding support."""

from typing import Any
from uuid import UUID

from mcp.types import Tool

from ..errors import EmbeddingError, NotFoundError, StoreConnectionError, ToolError, ValidationError


def get_tools() -> list[Tool]:
    """Get main store tools (full CRUD with embedding)."""
    return [
        Tool(
            name="store.add",
            description="Add a record to the main store (content and compressions). Automatically generates embedding from content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The knowledge content text",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["external", "internal"],
                        "description": "Origin: 'external' (from Zotero) or 'internal' (system-generated compression)",
                    },
                    "compression_level": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "0 = original content, 1+ = compression depth",
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "description": "UUIDs this was derived from (for compressions)",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata (extraction info, compression details, etc.)",
                    },
                    "zotero_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8,
                        "description": "8-char Zotero key (required if source_type='external')",
                    },
                    "language_code": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 5,
                        "description": "ISO 639-1 language code (e.g., 'en', 'es', 'zh')",
                    },
                },
                "required": ["content", "source_type"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="store.get",
            description="Retrieve a store record by UUID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the store record",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="store.update",
            description="Update a store record. Regenerates embedding if content changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the store record to update",
                    },
                    "content": {
                        "type": "string",
                        "description": "Updated content (will regenerate embedding)",
                    },
                    "compression_level": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Updated compression level",
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "description": "Updated source UUIDs",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Updated metadata",
                    },
                    "language_code": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 5,
                        "description": "Updated ISO 639-1 language code",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="store.delete",
            description="Delete a store record. Archives to forgotten_store with required reason.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the store record to delete",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Required: why this is being forgotten (stored in forgotten_store)",
                    },
                },
                "required": ["id", "reason"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="store.search",
            description="Search store records using Elasticsearch query DSL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": "Elasticsearch query DSL object (e.g., {'match': {'content': 'machine learning'}})",
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
    """Handle store tool calls."""
    from core.stores.schema import SourceType, StoreRecord

    es_stores = stores.get("es")
    if not es_stores:
        raise StoreConnectionError("elasticsearch", "Elasticsearch stores not initialized")

    main_store = es_stores.store

    if name == "store.add":
        # Generate embedding if service available
        embedding_model = None
        if embedding_service:
            try:
                await embedding_service.embed(arguments["content"])
                embedding_model = embedding_service.model
            except Exception as e:
                raise EmbeddingError(str(e), embedding_service.provider_name if embedding_service else None)

        # Parse source_type
        source_type = SourceType(arguments["source_type"])

        # Validate external sources have zotero_key
        if source_type == SourceType.EXTERNAL and not arguments.get("zotero_key"):
            raise ValidationError("zotero_key", "Required for external source_type")

        # Parse source_ids if provided
        source_ids = [UUID(sid) for sid in arguments.get("source_ids", [])]

        record = StoreRecord(
            source_type=source_type,
            content=arguments["content"],
            compression_level=arguments.get("compression_level", 0),
            source_ids=source_ids,
            metadata=arguments.get("metadata", {}),
            zotero_key=arguments.get("zotero_key"),
            language_code=arguments.get("language_code"),
            embedding_model=embedding_model,
        )

        record_id = await main_store.add(record)
        return {"id": str(record_id), "success": True}

    elif name == "store.get":
        record_id = UUID(arguments["id"])
        record = await main_store.get(record_id)
        if record is None:
            raise NotFoundError("store", arguments["id"])
        return record.model_dump(mode="json")

    elif name == "store.update":
        record_id = UUID(arguments["id"])

        # Build updates dict
        updates: dict[str, Any] = {}

        if "content" in arguments:
            updates["content"] = arguments["content"]
            # Regenerate embedding if content changes
            if embedding_service:
                try:
                    await embedding_service.embed(arguments["content"])
                    updates["embedding_model"] = embedding_service.model
                except Exception as e:
                    raise EmbeddingError(str(e), embedding_service.provider_name if embedding_service else None)

        if "compression_level" in arguments:
            updates["compression_level"] = arguments["compression_level"]

        if "source_ids" in arguments:
            updates["source_ids"] = [str(UUID(sid)) for sid in arguments["source_ids"]]

        if "metadata" in arguments:
            updates["metadata"] = arguments["metadata"]

        if "language_code" in arguments:
            updates["language_code"] = arguments["language_code"]

        if not updates:
            raise ValidationError("updates", "At least one field must be provided for update")

        success = await main_store.update(record_id, updates)
        if not success:
            raise NotFoundError("store", arguments["id"])

        return {"id": arguments["id"], "success": True}

    elif name == "store.delete":
        record_id = UUID(arguments["id"])
        reason = arguments["reason"]

        success = await main_store.delete(record_id, reason=reason)
        if not success:
            raise NotFoundError("store", arguments["id"])

        return {"id": arguments["id"], "deleted": True, "archived_to": "forgotten"}

    elif name == "store.search":
        query = arguments["query"]
        size = arguments.get("size", 10)
        records = await main_store.search(query, size=size)
        return {
            "count": len(records),
            "results": [r.model_dump(mode="json") for r in records],
        }

    else:
        raise ToolError(f"Unknown store tool: {name}")
