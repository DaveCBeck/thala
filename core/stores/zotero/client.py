"""
Async Zotero wrapper for local CRUD operations.

Communicates with zotero-local-crud plugin via HTTP on port 23119.
Uses httpx for native async HTTP requests.
"""

import logging
from typing import Any, Optional

import httpx

from ..schema import BaseRecord, SourceType, _utc_now
from .schemas import (
    ZoteroHealthStatus,
    ZoteroItem,
    ZoteroItemCreate,
    ZoteroItemUpdate,
    ZoteroSearchCondition,
    ZoteroSearchResult,
)

logger = logging.getLogger(__name__)


class ZoteroStore:
    """
    Async-safe Zotero client for local CRUD operations.

    Communicates with zotero-local-crud plugin running in Zotero.
    Port 23119 is the Zotero connector HTTP server (localhost-only).

    Example:
        async with ZoteroStore() as store:
            # Create an item
            key = await store.add(ZoteroItemCreate(
                itemType="book",
                fields={"title": "Example Book", "date": "2024"}
            ))

            # Read the item
            item = await store.get(key)

            # Search for items
            results = await store.quicksearch("example")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 23119,
        timeout: float = 30.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy init)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ZoteroStore":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ==================== CRUD Operations ====================

    async def add(self, item: ZoteroItemCreate) -> str:
        """
        Create a new item in Zotero.

        Args:
            item: ZoteroItemCreate with item type, fields, creators, etc.

        Returns:
            The 8-character Zotero key of the created item.

        Raises:
            httpx.HTTPStatusError: On API error
        """
        client = await self._get_client()

        # Convert Pydantic models to dicts, handling tags
        tags_data = []
        for tag in item.tags:
            if isinstance(tag, str):
                tags_data.append(tag)
            else:
                tags_data.append(tag.model_dump())

        payload = {
            "itemType": item.itemType,
            "fields": item.fields,
            "creators": [c.model_dump(exclude_none=True) for c in item.creators],
            "tags": tags_data,
            "collections": item.collections,
        }

        response = await client.post("/local-crud/items", json=payload)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"Created Zotero item with key: {result['key']}")
        return result["key"]

    async def get(self, zotero_key: str) -> Optional[ZoteroItem]:
        """
        Get a Zotero item by its key.

        Args:
            zotero_key: 8-character Zotero item key

        Returns:
            ZoteroItem if found, None if not found
        """
        client = await self._get_client()

        response = await client.post(
            "/local-crud/item",
            json={"action": "get", "key": zotero_key},
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        # Check for error in response body
        if "error" in data and data["error"] == "Item not found":
            return None

        return ZoteroItem.model_validate(data)

    async def update(self, zotero_key: str, updates: ZoteroItemUpdate) -> bool:
        """
        Update a Zotero item.

        Args:
            zotero_key: 8-character Zotero item key
            updates: ZoteroItemUpdate with fields to update

        Returns:
            True if updated, False if not found
        """
        client = await self._get_client()

        payload = {"action": "update", "key": zotero_key}
        if updates.fields is not None:
            payload["fields"] = updates.fields
        if updates.creators is not None:
            payload["creators"] = [
                c.model_dump(exclude_none=True) for c in updates.creators
            ]
        if updates.tags is not None:
            tags_data = []
            for tag in updates.tags:
                if isinstance(tag, str):
                    tags_data.append(tag)
                else:
                    tags_data.append(tag.model_dump())
            payload["tags"] = tags_data
        if updates.collections is not None:
            payload["collections"] = updates.collections

        response = await client.post("/local-crud/item", json=payload)

        if response.status_code == 404:
            return False

        data = response.json()
        if "error" in data and data["error"] == "Item not found":
            return False

        response.raise_for_status()
        logger.debug(f"Updated Zotero item: {zotero_key}")
        return True

    async def delete(self, zotero_key: str) -> bool:
        """
        Delete a Zotero item.

        Args:
            zotero_key: 8-character Zotero item key

        Returns:
            True if deleted, False if not found
        """
        client = await self._get_client()

        response = await client.post(
            "/local-crud/item",
            json={"action": "delete", "key": zotero_key},
        )

        if response.status_code == 404:
            return False

        # Check for error in response body
        if response.status_code == 200:
            data = response.json()
            if "error" in data and data["error"] == "Item not found":
                return False

        # 204 No Content is success
        if response.status_code == 204:
            logger.debug(f"Deleted Zotero item: {zotero_key}")
            return True

        response.raise_for_status()
        return True

    async def exists(self, zotero_key: str) -> bool:
        """Check if a Zotero item exists."""
        item = await self.get(zotero_key)
        return item is not None

    # ==================== Search ====================

    async def search(
        self,
        conditions: list[ZoteroSearchCondition],
        limit: int = 100,
        include_full_data: bool = False,
    ) -> list[ZoteroSearchResult | ZoteroItem]:
        """
        Search for Zotero items.

        Args:
            conditions: List of search conditions
            limit: Maximum results to return
            include_full_data: If True, return full ZoteroItem objects

        Returns:
            List of ZoteroSearchResult (or ZoteroItem if include_full_data=True)
        """
        client = await self._get_client()

        response = await client.post(
            "/local-crud/search",
            json={
                "conditions": [c.model_dump() for c in conditions],
                "limit": limit,
                "includeFullData": include_full_data,
            },
        )
        response.raise_for_status()

        data = response.json()

        if include_full_data:
            return [ZoteroItem.model_validate(item) for item in data.get("items", [])]
        else:
            return [
                ZoteroSearchResult.model_validate(item)
                for item in data.get("items", [])
            ]

    async def search_by_title(
        self, title: str, limit: int = 100
    ) -> list[ZoteroSearchResult]:
        """Convenience method to search by title."""
        return await self.search(
            conditions=[
                ZoteroSearchCondition(
                    condition="title", operator="contains", value=title
                )
            ],
            limit=limit,
        )

    async def search_by_tag(self, tag: str, limit: int = 100) -> list[ZoteroSearchResult]:
        """Convenience method to search by tag."""
        return await self.search(
            conditions=[ZoteroSearchCondition(condition="tag", operator="is", value=tag)],
            limit=limit,
        )

    async def quicksearch(self, query: str, limit: int = 100) -> list[ZoteroSearchResult]:
        """
        Quicksearch across all fields (like the Zotero search bar).

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching items
        """
        return await self.search(
            conditions=[
                ZoteroSearchCondition(
                    condition="quicksearch-everything",
                    operator="contains",
                    value=query,
                )
            ],
            limit=limit,
        )

    async def get_all(self, limit: int = 1000) -> list[ZoteroSearchResult]:
        """
        Get all items in the library.

        Args:
            limit: Maximum results to return

        Returns:
            List of all items (excluding attachments and notes)
        """
        return await self.search(conditions=[], limit=limit)

    # ==================== Health Check ====================

    async def health_check(self) -> ZoteroHealthStatus:
        """
        Check if the Zotero plugin is reachable and responsive.

        Returns:
            ZoteroHealthStatus with connection info
        """
        try:
            client = await self._get_client()
            response = await client.get("/local-crud/ping")

            if response.status_code == 200:
                data = response.json()
                return ZoteroHealthStatus(
                    healthy=True,
                    status=data.get("status"),
                    plugin=data.get("plugin"),
                    version=data.get("version"),
                    zoteroVersion=data.get("zoteroVersion"),
                    libraryID=data.get("libraryID"),
                )
            else:
                return ZoteroHealthStatus(
                    healthy=False, error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Zotero health check failed: {e}")
            return ZoteroHealthStatus(healthy=False, error=str(e))

    # ==================== BaseRecord Integration ====================

    async def get_by_record(self, record: BaseRecord) -> Optional[ZoteroItem]:
        """
        Get a Zotero item linked to a BaseRecord.

        Args:
            record: Any record with a zotero_key field

        Returns:
            ZoteroItem if found and zotero_key is set, None otherwise
        """
        if record.zotero_key is None:
            return None
        return await self.get(record.zotero_key)

    async def link_record(self, record: BaseRecord, zotero_key: str) -> BaseRecord:
        """
        Link a BaseRecord to an existing Zotero item.

        Args:
            record: The record to update
            zotero_key: The Zotero item key to link

        Returns:
            Updated record with zotero_key set

        Raises:
            ValueError: If the Zotero item doesn't exist
        """
        if not await self.exists(zotero_key):
            raise ValueError(f"Zotero item with key '{zotero_key}' not found")

        record.zotero_key = zotero_key
        record.source_type = SourceType.EXTERNAL
        record.updated_at = _utc_now()
        return record

    async def create_from_record(
        self,
        record: BaseRecord,
        item_type: str = "document",
        title: Optional[str] = None,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Create a Zotero item from a BaseRecord and link them.

        Args:
            record: The record to create a Zotero item for
            item_type: Zotero item type (default: "document")
            title: Title for the Zotero item
            extra_fields: Additional fields to set

        Returns:
            The Zotero key of the created item

        Side effects:
            Updates record.zotero_key and record.source_type
        """
        fields = extra_fields or {}
        if title:
            fields["title"] = title

        # Add record UUID to extra field for traceability
        extra = fields.get("extra", "")
        if extra:
            extra += "\n"
        extra += f"thala-id: {record.id}"
        fields["extra"] = extra

        zotero_key = await self.add(
            ZoteroItemCreate(
                itemType=item_type,
                fields=fields,
            )
        )

        record.zotero_key = zotero_key
        record.source_type = SourceType.EXTERNAL
        record.updated_at = _utc_now()

        logger.debug(f"Created Zotero item {zotero_key} for record {record.id}")
        return zotero_key
