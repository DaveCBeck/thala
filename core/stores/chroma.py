"""
Async ChromaDB wrapper for vector storage.

ChromaDB 1.0.0 has no native async - all calls wrapped with asyncio.to_thread().
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Optional
from uuid import UUID

import chromadb
from chromadb.config import Settings

from .schema import BaseRecord, WhoIWasRecord

if TYPE_CHECKING:
    from .elasticsearch import ElasticsearchStores

logger = logging.getLogger(__name__)


class ChromaStore:
    """Async-safe ChromaDB client for knowledge base vectors."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "knowledge",
        es_stores: Optional["ElasticsearchStores"] = None,
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
        self._es_stores = es_stores

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

    async def update(
        self,
        record: BaseRecord,
        embedding: list[float],
        document: str,
        reason: str = "Updated via API",
    ) -> UUID:
        """
        Update a record with history tracking.

        Requires es_stores to be configured for mandatory archiving to who_i_was.

        Args:
            record: Pydantic record (must have id)
            embedding: Vector embedding
            document: Text content for the document
            reason: Optional reason for the update

        Returns:
            The record's UUID

        Raises:
            RuntimeError: If es_stores not configured (archiving is mandatory)
        """
        if self._es_stores is None:
            raise RuntimeError(
                "ChromaStore.update() requires es_stores for mandatory archiving. "
                "Use add() for new records that don't require history tracking."
            )

        # Get existing record first
        existing = await self.get(record.id)

        if existing is not None:
            # Save snapshot to who_i_was (metadata + document)
            who_i_was = WhoIWasRecord(
                supersedes=record.id,
                reason=reason,
                previous_data={
                    "metadata": existing["metadata"],
                    "document": existing["document"],
                },
                original_store="top_of_mind",
            )
            await self._es_stores.who_i_was.add(who_i_was)

        # Perform upsert via add()
        return await self.add(record, embedding, document)

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

        if len(results["ids"]) == 0:
            return None

        return {
            "id": UUID(results["ids"][0]),
            "embedding": results["embeddings"][0] if results["embeddings"] is not None and len(results["embeddings"]) > 0 else None,
            "metadata": results["metadatas"][0] if results["metadatas"] is not None and len(results["metadatas"]) > 0 else None,
            "document": results["documents"][0] if results["documents"] is not None and len(results["documents"]) > 0 else None,
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

    async def delete(
        self,
        record_id: UUID,
        reason: str = "Deleted via API",
    ) -> bool:
        """
        Delete a record by UUID with mandatory history tracking.

        Requires es_stores to be configured for mandatory archiving to who_i_was.

        Args:
            record_id: UUID of record to delete
            reason: Optional reason for the deletion

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If es_stores not configured (archiving is mandatory)
        """
        if self._es_stores is None:
            raise RuntimeError(
                "ChromaStore.delete() requires es_stores for mandatory archiving."
            )

        # Check if exists first
        existing = await self.get(record_id)
        if existing is None:
            return False

        # Save snapshot to who_i_was (mandatory)
        who_i_was = WhoIWasRecord(
            supersedes=record_id,
            reason=reason,
            previous_data={
                "metadata": existing["metadata"],
                "document": existing["document"],
            },
            original_store="top_of_mind",
        )
        await self._es_stores.who_i_was.add(who_i_was)

        # Then delete from Chroma
        collection = await self._get_collection()
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
