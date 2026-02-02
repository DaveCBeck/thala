"""Protocol definition for ZoteroStore to ensure mock fidelity.

This protocol defines the interface that both real ZoteroStore and test mocks
must implement. It ensures Liskov Substitution compliance and prevents mock drift.

Usage:
    from core.stores.zotero.protocol import ZoteroStoreProtocol

    def __init__(self, zotero: ZoteroStoreProtocol | None = None):
        self._zotero = zotero
"""

from typing import Protocol

from .schemas import (
    ZoteroHealthStatus,
    ZoteroItem,
    ZoteroItemCreate,
    ZoteroItemUpdate,
    ZoteroSearchCondition,
    ZoteroSearchResult,
)


class ZoteroStoreProtocol(Protocol):
    """Protocol defining the ZoteroStore interface.

    Both real ZoteroStore and test mocks should conform to this protocol.
    This ensures Liskov Substitution compliance and prevents mock drift.

    The protocol captures the essential CRUD and search operations needed
    for store integration testing.
    """

    async def add(self, item: ZoteroItemCreate) -> str:
        """Create a new item in Zotero.

        Args:
            item: ZoteroItemCreate with item type, fields, creators, etc.

        Returns:
            The 8-character Zotero key of the created item.
        """
        ...

    async def get(self, zotero_key: str) -> ZoteroItem | None:
        """Get a Zotero item by its key.

        Args:
            zotero_key: 8-character Zotero item key

        Returns:
            ZoteroItem if found, None if not found
        """
        ...

    async def update(self, zotero_key: str, updates: ZoteroItemUpdate) -> bool:
        """Update a Zotero item.

        Args:
            zotero_key: 8-character Zotero item key
            updates: ZoteroItemUpdate with fields to update

        Returns:
            True if updated, False if not found
        """
        ...

    async def delete(self, zotero_key: str) -> bool:
        """Delete a Zotero item.

        Args:
            zotero_key: 8-character Zotero item key

        Returns:
            True if deleted, False if not found
        """
        ...

    async def search(
        self,
        conditions: list[ZoteroSearchCondition],
        limit: int = 100,
        include_full_data: bool = False,
    ) -> list[ZoteroSearchResult | ZoteroItem]:
        """Search for Zotero items.

        Args:
            conditions: List of search conditions
            limit: Maximum results to return
            include_full_data: If True, return full ZoteroItem objects

        Returns:
            List of ZoteroSearchResult (or ZoteroItem if include_full_data=True)
        """
        ...

    async def quicksearch(
        self,
        query: str,
        limit: int = 100,
    ) -> list[ZoteroSearchResult]:
        """Quicksearch across all fields (like the Zotero search bar).

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching items
        """
        ...

    async def health_check(self) -> ZoteroHealthStatus:
        """Check if the Zotero plugin is reachable and responsive.

        Returns:
            ZoteroHealthStatus with connection info
        """
        ...

    async def close(self) -> None:
        """Close connections and cleanup resources."""
        ...
