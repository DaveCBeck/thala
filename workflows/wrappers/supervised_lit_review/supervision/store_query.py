"""Store query service for supervision loops.

Provides dynamic access to Elasticsearch store for detailed paper content
at different compression levels.
"""

import logging
from typing import Any, Optional

from langchain_tools.base import get_store_manager, StoreManager
from core.stores import ZoteroStore

logger = logging.getLogger(__name__)


class SupervisionStoreQuery:
    """Query service for supervision loops to access detailed paper content.

    Enables loops to dynamically fetch paper content at different compression levels:
    - L0: Full original document (large)
    - L1: Short summary (~100 words)
    - L2: 10:1 compressed summary (medium)

    Queries ES directly by zotero_key without requiring paper_summaries.

    Usage:
        store_query = SupervisionStoreQuery()
        content, metadata = await store_query.get_paper_content("ABC12345")
    """

    def __init__(self):
        """Initialize store query service."""
        self._store_manager: Optional[StoreManager] = None
        self._zotero_client: Optional[ZoteroStore] = None

    @property
    def store_manager(self) -> StoreManager:
        """Get store manager (lazy initialization)."""
        if self._store_manager is None:
            self._store_manager = get_store_manager()
        return self._store_manager

    @property
    def zotero_client(self) -> ZoteroStore:
        """Get Zotero client (lazy initialization)."""
        if self._zotero_client is None:
            self._zotero_client = ZoteroStore()
        return self._zotero_client

    async def close(self):
        """Close any open connections."""
        if self._zotero_client is not None:
            await self._zotero_client.close()
            self._zotero_client = None

    async def get_paper_content(
        self,
        zotero_key: str,
        compression_level: int = 2,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Fetch paper content from store by zotero_key.

        Queries ES directly by zotero_key field, trying the specified
        compression level first, then falling back to alternatives.

        Args:
            zotero_key: 8-char Zotero citation key
            compression_level: 0=full, 1=short summary, 2=10:1 summary

        Returns:
            Tuple of (content, metadata) or (None, None) if not found
        """
        store = self.store_manager.es_stores.store

        try:
            # Query ES by zotero_key at specified compression level
            query = {"term": {"zotero_key": zotero_key}}
            records = await store.search(
                query=query,
                size=1,
                compression_level=compression_level,
            )

            if records:
                record = records[0]
                return record.content, record.metadata

            # Try alternative compression levels
            for alt_level in [1, 2, 0]:
                if alt_level == compression_level:
                    continue
                records = await store.search(
                    query=query,
                    size=1,
                    compression_level=alt_level,
                )
                if records:
                    logger.debug(
                        f"Found {zotero_key} at L{alt_level} instead of L{compression_level}"
                    )
                    record = records[0]
                    return record.content, record.metadata

            logger.debug(f"No ES record found for zotero_key: {zotero_key}")
            return None, None

        except Exception as e:
            logger.warning(f"Error fetching content for {zotero_key}: {e}")
            return None, None

    async def get_zotero_metadata(self, zotero_key: str) -> Optional[dict[str, Any]]:
        """Fetch paper metadata from Zotero as fallback.

        Args:
            zotero_key: 8-char Zotero citation key

        Returns:
            Dict with title, authors, year, etc. or None if not found
        """
        try:
            item = await self.zotero_client.get(zotero_key)
            if item:
                # Extract author names from creator dicts
                authors = []
                for c in item.creators or []:
                    if c.get("name"):
                        authors.append(c["name"])
                    elif c.get("lastName"):
                        first = c.get("firstName", "")
                        last = c.get("lastName", "")
                        authors.append(f"{first} {last}".strip())

                # Extract year from date field (format varies: "2024", "2024-01-15", etc.)
                date_str = item.fields.get("date", "")
                year = date_str[:4] if date_str and len(date_str) >= 4 else None

                return {
                    "title": item.fields.get("title"),
                    "authors": authors,
                    "year": year,
                    "doi": item.fields.get("DOI"),
                    "item_type": item.itemType,
                }
            return None
        except Exception as e:
            logger.warning(f"Error fetching Zotero metadata for {zotero_key}: {e}")
            return None
