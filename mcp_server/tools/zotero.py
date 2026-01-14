"""zotero tools - Full CRUD for external content management."""

from typing import Any

from mcp.types import Tool

from ..errors import NotFoundError, ToolError
from ..response_utils import format_search_results, format_single_record
from ..store_utils import get_zotero_store


def get_tools() -> list[Tool]:
    """Get Zotero tools (full CRUD)."""
    return [
        Tool(
            name="zotero.add",
            description="Create a new item in Zotero.",
            inputSchema={
                "type": "object",
                "properties": {
                    "itemType": {
                        "type": "string",
                        "description": "Zotero item type (book, journalArticle, webpage, document, note, etc.)",
                    },
                    "fields": {
                        "type": "object",
                        "description": "Item fields (title, date, url, abstractNote, etc.)",
                    },
                    "creators": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "firstName": {"type": "string"},
                                "lastName": {"type": "string"},
                                "name": {
                                    "type": "string",
                                    "description": "For single-field names",
                                },
                                "creatorType": {"type": "string", "default": "author"},
                            },
                        },
                        "description": "List of creators (authors, editors, etc.)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags",
                    },
                },
                "required": ["itemType"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="zotero.get",
            description="Get a Zotero item by its 8-character key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "zotero_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8,
                        "description": "8-character Zotero item key",
                    },
                },
                "required": ["zotero_key"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="zotero.update",
            description="Update a Zotero item.",
            inputSchema={
                "type": "object",
                "properties": {
                    "zotero_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8,
                        "description": "8-character Zotero item key",
                    },
                    "fields": {
                        "type": "object",
                        "description": "Fields to update (title, date, url, etc.)",
                    },
                    "creators": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Updated creators list (replaces existing)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Updated tags list (replaces existing)",
                    },
                },
                "required": ["zotero_key"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="zotero.delete",
            description="Delete a Zotero item.",
            inputSchema={
                "type": "object",
                "properties": {
                    "zotero_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 8,
                        "description": "8-character Zotero item key",
                    },
                },
                "required": ["zotero_key"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="zotero.search",
            description="Advanced search in Zotero with conditions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition": {
                                    "type": "string",
                                    "description": "Field to search (title, tag, creator, etc.)",
                                },
                                "operator": {
                                    "type": "string",
                                    "default": "contains",
                                    "description": "Operator (is, contains, doesNotContain, etc.)",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Search value",
                                },
                                "required": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Whether this condition is required",
                                },
                            },
                            "required": ["condition", "value"],
                        },
                        "description": "Search conditions",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100,
                        "description": "Maximum results to return",
                    },
                },
                "required": ["conditions"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="zotero.quicksearch",
            description="Quick search across all Zotero fields (like the search bar).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100,
                        "description": "Maximum results to return",
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
    """Handle Zotero tool calls."""
    from core.stores.zotero import (
        ZoteroCreator,
        ZoteroItemCreate,
        ZoteroItemUpdate,
        ZoteroSearchCondition,
    )

    zotero_store = get_zotero_store(stores)

    if name == "zotero.add":
        creators = []
        for c in arguments.get("creators", []):
            creators.append(
                ZoteroCreator(
                    firstName=c.get("firstName"),
                    lastName=c.get("lastName"),
                    name=c.get("name"),
                    creatorType=c.get("creatorType", "author"),
                )
            )

        item = ZoteroItemCreate(
            itemType=arguments["itemType"],
            fields=arguments.get("fields", {}),
            creators=creators,
            tags=arguments.get("tags", []),
        )

        zotero_key = await zotero_store.add(item)
        return {"zotero_key": zotero_key, "success": True}

    elif name == "zotero.get":
        zotero_key = arguments["zotero_key"]
        item = await zotero_store.get(zotero_key)
        if item is None:
            raise NotFoundError("zotero", zotero_key)
        return format_single_record(item)

    elif name == "zotero.update":
        zotero_key = arguments["zotero_key"]

        creators = None
        if "creators" in arguments:
            creators = [
                ZoteroCreator(
                    firstName=c.get("firstName"),
                    lastName=c.get("lastName"),
                    name=c.get("name"),
                    creatorType=c.get("creatorType", "author"),
                )
                for c in arguments["creators"]
            ]

        updates = ZoteroItemUpdate(
            fields=arguments.get("fields"),
            creators=creators,
            tags=arguments.get("tags"),
        )

        success = await zotero_store.update(zotero_key, updates)
        if not success:
            raise NotFoundError("zotero", zotero_key)
        return {"zotero_key": zotero_key, "success": True}

    elif name == "zotero.delete":
        zotero_key = arguments["zotero_key"]
        success = await zotero_store.delete(zotero_key)
        if not success:
            raise NotFoundError("zotero", zotero_key)
        return {"zotero_key": zotero_key, "deleted": True}

    elif name == "zotero.search":
        conditions = [
            ZoteroSearchCondition(
                condition=c["condition"],
                operator=c.get("operator", "contains"),
                value=c["value"],
                required=c.get("required", True),
            )
            for c in arguments["conditions"]
        ]
        limit = arguments.get("limit", 100)

        results = await zotero_store.search(conditions, limit=limit)
        return format_search_results(results)

    elif name == "zotero.quicksearch":
        query = arguments["query"]
        limit = arguments.get("limit", 100)
        results = await zotero_store.quicksearch(query, limit=limit)
        return format_search_results(results)

    else:
        raise ToolError(f"Unknown zotero tool: {name}")
