"""top_of_mind tools - Full CRUD for vector store with semantic search."""

from typing import Any

from mcp.types import Tool

from ..embedding_utils import validate_embedding_available
from ..errors import EmbeddingError, NotFoundError, ToolError, ValidationError
from ..store_utils import get_chroma_store
from ..validation_utils import parse_uuid_arg


def get_tools() -> list[Tool]:
    """Get top_of_mind tools (full CRUD with embedding and semantic search)."""
    return [
        Tool(
            name="top_of_mind.add",
            description="Add a new record to top_of_mind vector store. Automatically generates embedding from content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text content to embed and store",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["external", "internal"],
                        "description": "Origin: 'external' (has zotero_key) or 'internal' (system-generated)",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata",
                    },
                    "zotero_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8,
                        "description": "8-char Zotero key (required if source_type='external')",
                    },
                },
                "required": ["content", "source_type"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="top_of_mind.get",
            description="Retrieve a record by UUID from top_of_mind.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the record",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="top_of_mind.update",
            description="Update a record in top_of_mind. Automatically archives previous version to who_i_was and regenerates embedding.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the record to update",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (will regenerate embedding)",
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
                "required": ["id", "content", "reason"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="top_of_mind.delete",
            description="Delete a record from top_of_mind. Archives to who_i_was before deletion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the record to delete",
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
            name="top_of_mind.search",
            description="Semantic search in top_of_mind using vector similarity. Query is embedded and compared against stored vectors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (will be embedded for similarity search)",
                    },
                    "n_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Number of results to return",
                    },
                    "where": {
                        "type": "object",
                        "description": "Optional ChromaDB metadata filter (e.g., {'source_type': 'external'})",
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
    """Handle top_of_mind tool calls."""
    from core.stores.schema import BaseRecord, SourceType

    chroma_store = get_chroma_store(stores)

    if name in ("top_of_mind.add", "top_of_mind.update", "top_of_mind.search"):
        await validate_embedding_available(embedding_service)

    if name == "top_of_mind.add":
        content = arguments["content"]

        try:
            embedding = await embedding_service.embed(content)
        except Exception as e:
            raise EmbeddingError(str(e), embedding_service.provider_name)

        source_type = SourceType(arguments["source_type"])

        if source_type == SourceType.EXTERNAL and not arguments.get("zotero_key"):
            raise ValidationError("zotero_key", "Required for external source_type")

        record = BaseRecord(
            source_type=source_type,
            content=content,
            metadata=arguments.get("metadata", {}),
            zotero_key=arguments.get("zotero_key"),
            embedding_model=embedding_service.model,
        )

        record_id = await chroma_store.add(
            record=record,
            embedding=embedding,
            document=content,
        )

        return {"id": str(record_id), "success": True}

    elif name == "top_of_mind.get":
        record_id = parse_uuid_arg(arguments)
        result = await chroma_store.get(record_id)
        if result is None:
            raise NotFoundError("top_of_mind", arguments["id"])

        return {
            "id": str(result["id"]),
            "document": result["document"],
            "metadata": result["metadata"],
            "has_embedding": result["embedding"] is not None,
        }

    elif name == "top_of_mind.update":
        record_id = parse_uuid_arg(arguments)
        content = arguments["content"]
        reason = arguments["reason"]

        existing = await chroma_store.get(record_id)
        if existing is None:
            raise NotFoundError("top_of_mind", arguments["id"])

        try:
            embedding = await embedding_service.embed(content)
        except Exception as e:
            raise EmbeddingError(str(e), embedding_service.provider_name)

        metadata = existing.get("metadata", {})
        if "metadata" in arguments:
            metadata.update(arguments["metadata"])

        source_type_str = metadata.get("source_type", "internal")
        source_type = SourceType(source_type_str)

        record = BaseRecord(
            id=record_id,
            source_type=source_type,
            content=content,
            metadata=metadata,
            zotero_key=metadata.get("zotero_key"),
            embedding_model=embedding_service.model,
        )

        await chroma_store.update(
            record=record,
            embedding=embedding,
            document=content,
            reason=reason,
        )

        return {"id": arguments["id"], "success": True}

    elif name == "top_of_mind.delete":
        record_id = parse_uuid_arg(arguments)
        reason = arguments["reason"]

        success = await chroma_store.delete(record_id, reason=reason)
        if not success:
            raise NotFoundError("top_of_mind", arguments["id"])

        return {"id": arguments["id"], "deleted": True, "archived_to": "who_i_was"}

    elif name == "top_of_mind.search":
        query = arguments["query"]
        n_results = arguments.get("n_results", 10)
        where = arguments.get("where")

        try:
            query_embedding = await embedding_service.embed(query)
        except Exception as e:
            raise EmbeddingError(str(e), embedding_service.provider_name)

        results = await chroma_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
        )

        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "id": str(r["id"]),
                    "distance": r["distance"],
                    "document": r["document"],
                    "metadata": r["metadata"],
                }
                for r in results
            ],
        }

    else:
        raise ToolError(f"Unknown top_of_mind tool: {name}")
