"""
Async Elasticsearch wrapper for structured storage.

Uses elasticsearch[async] 8.17.0 with native async support.
Routes to different ES instances based on index.

Index structure:
- ES Coherence (9201): store_l0, store_l1, store_l2, coherence
- ES Forgotten (9200): who_i_was, forgotten
"""

import logging
from typing import Any, Optional, TypeVar
from uuid import UUID

from elasticsearch import AsyncElasticsearch, NotFoundError
from pydantic import BaseModel

from .schema import (
    BaseRecord,
    CoherenceRecord,
    ForgottenRecord,
    StoreRecord,
    WhoIWasRecord,
    _utc_now,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseRecord)


class ElasticsearchStores:
    """
    Manages all Elasticsearch-backed stores.

    Routes operations to correct ES instance:
    - Port 9201: store_l0, store_l1, store_l2, coherence
    - Port 9200: who_i_was, forgotten
    """

    def __init__(
        self,
        coherence_host: str = "http://localhost:9201",
        forgotten_host: str = "http://localhost:9200",
        request_timeout: int = 30,
    ):
        # ES instance for coherence and store indices
        self._coherence_client = AsyncElasticsearch(
            hosts=[coherence_host],
            request_timeout=request_timeout,
            max_retries=3,
            retry_on_timeout=True,
            http_compress=True,
        )

        # ES instance for forgotten and who_i_was
        self._forgotten_client = AsyncElasticsearch(
            hosts=[forgotten_host],
            request_timeout=request_timeout,
            max_retries=3,
            retry_on_timeout=True,
            http_compress=True,
        )

        # Store instances
        self.coherence = CoherenceStore(self._coherence_client, self)
        self.store = MainStore(self._coherence_client, self)
        self.who_i_was = WhoIWasStore(self._forgotten_client)
        self.forgotten = ForgottenStore(self._forgotten_client)

    async def close(self):
        """Close all client connections."""
        await self._coherence_client.close()
        await self._forgotten_client.close()

    async def health_check(self) -> dict[str, Any]:
        """Check health of all ES instances."""
        try:
            coherence = await self._coherence_client.cluster.health()
            forgotten = await self._forgotten_client.cluster.health()
            return {
                "coherence_instance": coherence["status"],
                "forgotten_instance": forgotten["status"],
                "healthy": all(
                    s in ("green", "yellow")
                    for s in (coherence["status"], forgotten["status"])
                ),
            }
        except Exception as e:
            logger.error(f"ES health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class BaseElasticsearchStore:
    """Base class for ES-backed stores."""

    index_name: str
    record_class: type[BaseRecord]

    def __init__(self, client: AsyncElasticsearch):
        self._client = client

    def _get_index_name(self, record: Optional[BaseRecord] = None) -> str:
        """Get the index name. Override in subclasses for dynamic routing."""
        return self.index_name

    async def add(self, record: T) -> UUID:
        """Add a record to the store."""
        index = self._get_index_name(record)
        await self._client.index(
            index=index,
            id=str(record.id),
            document=record.model_dump(mode="json"),
        )
        logger.debug(f"Added record {record.id} to {index}")
        return record.id

    async def get(self, record_id: UUID, index: Optional[str] = None) -> Optional[T]:
        """Get a record by UUID."""
        index = index or self.index_name
        try:
            response = await self._client.get(
                index=index,
                id=str(record_id),
            )
            return self.record_class.model_validate(response["_source"])
        except NotFoundError:
            return None

    async def update(self, record_id: UUID, updates: dict[str, Any], index: Optional[str] = None) -> bool:
        """
        Partially update a record.

        Args:
            record_id: UUID of record to update
            updates: Dict of fields to update
            index: Optional index name override

        Returns:
            True if updated, False if not found
        """
        index = index or self.index_name
        try:
            updates["updated_at"] = _utc_now().isoformat()
            await self._client.update(
                index=index,
                id=str(record_id),
                doc=updates,
            )
            logger.debug(f"Updated record {record_id} in {index}")
            return True
        except NotFoundError:
            return False

    async def delete(self, record_id: UUID, index: Optional[str] = None) -> bool:
        """Delete a record by UUID."""
        index = index or self.index_name
        try:
            await self._client.delete(
                index=index,
                id=str(record_id),
            )
            logger.debug(f"Deleted record {record_id} from {index}")
            return True
        except NotFoundError:
            return False

    async def search(
        self,
        query: dict[str, Any],
        size: int = 10,
        index: Optional[str] = None,
    ) -> list[T]:
        """
        Search for records.

        Args:
            query: Elasticsearch query DSL
            size: Max results to return
            index: Optional index name or pattern (e.g., "store_l*")

        Returns:
            List of matching records as Pydantic models
        """
        index = index or self.index_name
        response = await self._client.search(
            index=index,
            query=query,
            size=size,
        )

        return [
            self.record_class.model_validate(hit["_source"])
            for hit in response["hits"]["hits"]
        ]

    async def exists(self, record_id: UUID, index: Optional[str] = None) -> bool:
        """Check if a record exists."""
        index = index or self.index_name
        return await self._client.exists(index=index, id=str(record_id))


class CoherenceStore(BaseElasticsearchStore):
    """
    Store for identity, beliefs, preferences with confidence scores.

    Auto-versioning: update() automatically creates WhoIWasRecord.
    """

    index_name = "coherence"
    record_class = CoherenceRecord

    def __init__(self, client: AsyncElasticsearch, stores: "ElasticsearchStores"):
        super().__init__(client)
        self._stores = stores

    async def update(self, record_id: UUID, updates: dict[str, Any]) -> bool:
        """
        Update a coherence record with automatic versioning.

        Creates a WhoIWasRecord before updating to preserve history (full snapshot).
        """
        # Get current record before updating
        current = await self.get(record_id)
        if current is None:
            return False

        # Create WhoIWasRecord to preserve history with full snapshot
        who_i_was = WhoIWasRecord(
            supersedes=record_id,
            reason=updates.get("_change_reason", "Updated via API"),
            previous_data=current.model_dump(mode="json"),
            original_store="coherence",
        )

        # Remove internal fields from updates
        updates.pop("_change_reason", None)

        # Save to who_i_was store
        await self._stores.who_i_was.add(who_i_was)

        # Now perform the update
        updates["updated_at"] = _utc_now().isoformat()
        await self._client.update(
            index=self.index_name,
            id=str(record_id),
            doc=updates,
        )

        logger.debug(
            f"Updated coherence record {record_id}, "
            f"version saved as {who_i_was.id}"
        )
        return True

    async def delete(self, record_id: UUID, reason: str = "Deleted via API") -> bool:
        """
        Delete a coherence record with history preservation.

        Creates a WhoIWasRecord before deleting to preserve snapshot.
        """
        current = await self.get(record_id)
        if current is None:
            return False

        # Save snapshot to who_i_was before deleting
        who_i_was = WhoIWasRecord(
            supersedes=record_id,
            reason=reason,
            previous_data=current.model_dump(mode="json"),
            original_store="coherence",
        )
        await self._stores.who_i_was.add(who_i_was)

        # Then delete using parent class method
        try:
            await self._client.delete(
                index=self.index_name,
                id=str(record_id),
            )
            logger.debug(
                f"Deleted coherence record {record_id}, "
                f"snapshot saved as {who_i_was.id}"
            )
            return True
        except NotFoundError:
            return False


class MainStore(BaseElasticsearchStore):
    """
    Store for all relevant content - originals and compressions.

    Routes to different indices based on compression_level:
    - store_l0: Original documents (compression_level=0)
    - store_l1: Short summaries (compression_level=1)
    - store_l2: 10:1 summaries (compression_level=2)
    """

    index_name = "store_l0"  # Default for backwards compatibility
    record_class = StoreRecord

    # Index mapping by compression level
    COMPRESSION_INDICES = {
        0: "store_l0",
        1: "store_l1",
        2: "store_l2",
    }

    def __init__(self, client: AsyncElasticsearch, stores: "ElasticsearchStores"):
        super().__init__(client)
        self._stores = stores

    def _get_index_name(self, record: Optional[BaseRecord] = None) -> str:
        """Route to correct index based on compression_level."""
        if record is None:
            return self.index_name
        if isinstance(record, StoreRecord):
            return self.COMPRESSION_INDICES.get(record.compression_level, "store_l0")
        return self.index_name

    def _index_for_level(self, compression_level: int) -> str:
        """Get index name for a compression level."""
        return self.COMPRESSION_INDICES.get(compression_level, "store_l0")

    async def get(self, record_id: UUID, compression_level: Optional[int] = None) -> Optional[StoreRecord]:
        """
        Get a record by UUID.

        Args:
            record_id: UUID of the record
            compression_level: If known, speeds up lookup. Otherwise searches all store indices.
        """
        if compression_level is not None:
            index = self._index_for_level(compression_level)
            return await super().get(record_id, index=index)

        # Search across all store indices
        for level in self.COMPRESSION_INDICES.values():
            result = await super().get(record_id, index=level)
            if result:
                return result
        return None

    async def update(self, record_id: UUID, updates: dict[str, Any], compression_level: int = 0) -> bool:
        """
        Partially update a record.

        Args:
            record_id: UUID of record to update
            updates: Dict of fields to update
            compression_level: Which index to update in
        """
        index = self._index_for_level(compression_level)
        return await super().update(record_id, updates, index=index)

    async def delete(self, record_id: UUID, reason: str, compression_level: Optional[int] = None) -> bool:
        """
        Delete a store record with required reason, archiving to forgotten_store.

        Args:
            record_id: UUID of record to delete
            reason: Required explanation for why this is being forgotten
            compression_level: If known, speeds up lookup

        Returns:
            True if deleted, False if not found
        """
        current = await self.get(record_id, compression_level=compression_level)
        if current is None:
            return False

        index = self._index_for_level(current.compression_level)

        # Archive to forgotten_store with full snapshot
        await self._stores.forgotten.forget(
            current,
            reason,
            index,
            previous_data=current.model_dump(mode="json"),
        )

        # Then delete
        try:
            await self._client.delete(
                index=index,
                id=str(record_id),
            )
            logger.debug(f"Deleted store record {record_id} from {index}, archived to forgotten")
            return True
        except NotFoundError:
            return False

    async def search(
        self,
        query: dict[str, Any],
        size: int = 10,
        compression_level: Optional[int] = None,
    ) -> list[StoreRecord]:
        """
        Search for records.

        Args:
            query: Elasticsearch query DSL
            size: Max results to return
            compression_level: If specified, search only that level. Otherwise search all.

        Returns:
            List of matching StoreRecords
        """
        if compression_level is not None:
            index = self._index_for_level(compression_level)
        else:
            # Search across all store indices
            index = "store_l*"

        return await super().search(query, size, index=index)

    async def knn_search(
        self,
        embedding: list[float],
        k: int = 10,
        compression_level: Optional[int] = None,
        num_candidates: int = 100,
    ) -> list[tuple[StoreRecord, float]]:
        """
        Perform KNN vector search on summaries.

        Args:
            embedding: Query embedding vector
            k: Number of results to return
            compression_level: 1 or 2 (l0 has no embeddings). None searches both.
            num_candidates: Number of candidates to consider (higher = more accurate)

        Returns:
            List of (record, score) tuples sorted by similarity
        """
        if compression_level == 0:
            raise ValueError("store_l0 does not have embeddings - use text search")

        if compression_level is not None:
            index = self._index_for_level(compression_level)
        else:
            # Search l1 and l2 (both have embeddings)
            index = "store_l1,store_l2"

        response = await self._client.search(
            index=index,
            knn={
                "field": "embedding",
                "query_vector": embedding,
                "k": k,
                "num_candidates": num_candidates,
            },
        )

        results = []
        for hit in response["hits"]["hits"]:
            record = self.record_class.model_validate(hit["_source"])
            score = hit["_score"]
            results.append((record, score))

        return results


class WhoIWasStore(BaseElasticsearchStore):
    """Store for edit history - temporal records of superseded state."""

    index_name = "who_i_was"
    record_class = WhoIWasRecord

    async def get_history(self, record_id: UUID) -> list[WhoIWasRecord]:
        """Get all historical versions that superseded this record."""
        return await self.search(
            query={"term": {"supersedes": str(record_id)}},
            size=100,
        )


class ForgottenStore(BaseElasticsearchStore):
    """Store for archived/forgotten content."""

    index_name = "forgotten"
    record_class = ForgottenRecord

    async def forget(
        self,
        record: BaseRecord,
        reason: str,
        original_store: str,
        previous_data: Optional[dict] = None,
    ) -> UUID:
        """
        Archive a record to the forgotten store.

        Args:
            record: The record being forgotten
            reason: Why it's being forgotten
            original_store: Which store it came from ('coherence', 'store_l0', etc.)
            previous_data: Optional snapshot; defaults to record.model_dump()

        Returns:
            UUID of the forgotten record
        """
        forgotten = ForgottenRecord(
            source_type=record.source_type,
            zotero_key=record.zotero_key,
            forgotten_reason=reason,
            original_store=original_store,
            previous_data=previous_data or record.model_dump(mode="json"),
        )

        await self.add(forgotten)
        logger.info(f"Archived record to forgotten: {forgotten.id}, reason: {reason}")
        return forgotten.id
