"""
Shared utilities and store access for LangChain tools.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from core.stores.elasticsearch import ElasticsearchStores
from core.stores.chroma import ChromaStore
from core.stores.zotero import ZoteroStore
from core.embedding import EmbeddingService

load_dotenv()


class StoreManager:
    """
    Manages store connections for LangChain tools.

    Uses lazy initialization - stores are created on first access.
    Configurable via environment variables (same as MCP server).
    """

    _es_stores: Optional[ElasticsearchStores] = None
    _chroma: Optional[ChromaStore] = None
    _zotero: Optional[ZoteroStore] = None
    _embedding: Optional[EmbeddingService] = None

    def __init__(
        self,
        es_coherence_host: Optional[str] = None,
        es_forgotten_host: Optional[str] = None,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        zotero_host: Optional[str] = None,
        zotero_port: Optional[int] = None,
    ):
        """
        Initialize StoreManager with optional custom hosts.

        If not provided, uses environment variables or defaults.
        """
        self._es_coherence_host = es_coherence_host or os.environ.get(
            "THALA_ES_COHERENCE_HOST", "http://localhost:9201"
        )
        self._es_forgotten_host = es_forgotten_host or os.environ.get(
            "THALA_ES_FORGOTTEN_HOST", "http://localhost:9200"
        )
        self._chroma_host = chroma_host or os.environ.get(
            "THALA_CHROMA_HOST", "localhost"
        )
        self._chroma_port = chroma_port or int(os.environ.get(
            "THALA_CHROMA_PORT", "8000"
        ))
        self._zotero_host = zotero_host or os.environ.get(
            "THALA_ZOTERO_HOST", "localhost"
        )
        self._zotero_port = zotero_port or int(os.environ.get(
            "THALA_ZOTERO_PORT", "23119"
        ))

    @property
    def es_stores(self) -> ElasticsearchStores:
        """Get Elasticsearch stores (lazy init)."""
        if self._es_stores is None:
            self._es_stores = ElasticsearchStores(
                coherence_host=self._es_coherence_host,
                forgotten_host=self._es_forgotten_host,
            )
        return self._es_stores

    @property
    def chroma(self) -> ChromaStore:
        """Get ChromaDB store (lazy init)."""
        if self._chroma is None:
            self._chroma = ChromaStore(
                host=self._chroma_host,
                port=self._chroma_port,
                es_stores=self.es_stores,  # For history tracking
            )
        return self._chroma

    @property
    def zotero(self) -> ZoteroStore:
        """Get Zotero store (lazy init)."""
        if self._zotero is None:
            self._zotero = ZoteroStore(
                host=self._zotero_host,
                port=self._zotero_port,
            )
        return self._zotero

    @property
    def embedding(self) -> EmbeddingService:
        """Get embedding service (lazy init)."""
        if self._embedding is None:
            self._embedding = EmbeddingService()
        return self._embedding

    async def close(self):
        """Close all store connections."""
        if self._es_stores is not None:
            await self._es_stores.close()
        if self._embedding is not None:
            await self._embedding.close()
        if self._zotero is not None:
            await self._zotero.close()


# Singleton instance for convenience
_default_manager: Optional[StoreManager] = None


def get_store_manager() -> StoreManager:
    """Get the default StoreManager instance (creates if needed)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = StoreManager()
    return _default_manager
