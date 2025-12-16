"""
Async ChromaDB wrapper for vector storage.

ChromaDB 1.0.0 has no native async - all calls wrapped with asyncio.to_thread().
"""

import asyncio
import json
import logging
from typing import Optional
from uuid import UUID

import chromadb
from chromadb.config import Settings

from .schema import BaseRecord

logger = logging.getLogger(__name__)


class ChromaStore:
    """Async-safe ChromaDB client for knowledge base vectors."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "knowledge",
    ):
        self.collection_name = collection_name
        self._client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )
        self._collection: Optional[chromadb.Collection] = None

    async def _get_collection(self) -> chromadb.Collection:
        """Get or create the collection (lazy init)."""
        if self._collection is None:
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def add(
        self,
        record: BaseRecord,
        embedding: list[float],
        document: str,
    ) -> UUID:
        """
        Add a record with its embedding to the vector store.

        Args:
            record: Pydantic record (must have id, will use metadata)
            embedding: Vector embedding
            document: Text content for the document

        Returns:
            The record's UUID
        """
        collection = await self._get_collection()
        metadata = self._sanitize_metadata(record.model_dump(mode="json"))

        await asyncio.to_thread(
            collection.upsert,
            ids=[str(record.id)],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document],
        )

        logger.debug(f"Added/updated record {record.id} in Chroma")
        return record.id

    async def get(self, record_id: UUID) -> Optional[dict]:
        """
        Get a record by UUID.

        Returns:
            Dict with 'id', 'embedding', 'metadata', 'document' or None if not found
        """
        collection = await self._get_collection()

        results = await asyncio.to_thread(
            collection.get,
            ids=[str(record_id)],
            include=["embeddings", "metadatas", "documents"],
        )

        if not results["ids"]:
            return None

        return {
            "id": UUID(results["ids"][0]),
            "embedding": results["embeddings"][0] if results["embeddings"] else None,
            "metadata": results["metadatas"][0] if results["metadatas"] else None,
            "document": results["documents"][0] if results["documents"] else None,
        }

    async def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of dicts with 'id', 'distance', 'metadata', 'document'
        """
        collection = await self._get_collection()

        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        # Unwrap nested lists (ChromaDB returns nested even for single query)
        return [
            {
                "id": UUID(id_),
                "distance": distance,
                "metadata": metadata,
                "document": document,
            }
            for id_, distance, metadata, document in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
                results["documents"][0],
            )
        ]

    async def delete(self, record_id: UUID) -> bool:
        """
        Delete a record by UUID.

        Returns:
            True if deleted, False if not found
        """
        collection = await self._get_collection()

        # Check if exists first
        existing = await self.get(record_id)
        if existing is None:
            return False

        await asyncio.to_thread(collection.delete, ids=[str(record_id)])
        logger.debug(f"Deleted record {record_id} from Chroma")
        return True

    async def count(self) -> int:
        """Get total document count in collection."""
        collection = await self._get_collection()
        return await asyncio.to_thread(collection.count)

    async def health_check(self) -> bool:
        """Check if ChromaDB is reachable."""
        try:
            await asyncio.to_thread(self._client.heartbeat)
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """
        Clean metadata for ChromaDB storage.

        ChromaDB only supports: str, int, float, bool
        Complex types are serialized to JSON strings.
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, dict)):
                clean[key] = json.dumps(value)
            else:
                clean[key] = str(value)
        return clean
