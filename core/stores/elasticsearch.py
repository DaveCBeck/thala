"""
Async Elasticsearch wrapper for structured storage.

Uses elasticsearch[async] 8.17.0 with native async support.
Routes to different ES instances based on index.
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
    - Port 9201: coherence, store, who_i_was indices
    - Port 9200: forgotten index
    """

    def __init__(
        self,
        coherence_host: str = "http://localhost:9201",
        forgotten_host: str = "http://localhost:9200",
        request_timeout: int = 30,
    ):
        # ES instance for coherence, store, who_i_was
        self._coherence_client = AsyncElasticsearch(
            hosts=[coherence_host],
            request_timeout=request_timeout,
            max_retries=3,
            retry_on_timeout=True,
            http_compress=True,
        )

        # ES instance for forgotten
        self._forgotten_client = AsyncElasticsearch(
            hosts=[forgotten_host],
            request_timeout=request_timeout,
            max_retries=3,
            retry_on_timeout=True,
            http_compress=True,
        )

        # Index routing
        self._routes = {
            "coherence": self._coherence_client,
            "store": self._coherence_client,
            "who_i_was": self._coherence_client,
            "forgotten": self._forgotten_client,
        }

        # Store instances
        self.coherence = CoherenceStore(self._coherence_client, self)
        self.store = MainStore(self._coherence_client)
        self.who_i_was = WhoIWasStore(self._coherence_client)
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

    async def add(self, record: T) -> UUID:
        """Add a record to the store."""
        await self._client.index(
            index=self.index_name,
            id=str(record.id),
            document=record.model_dump(mode="json"),
        )
        logger.debug(f"Added record {record.id} to {self.index_name}")
        return record.id

    async def get(self, record_id: UUID) -> Optional[T]:
        """Get a record by UUID."""
        try:
            response = await self._client.get(
                index=self.index_name,
                id=str(record_id),
            )
            return self.record_class.model_validate(response["_source"])
        except NotFoundError:
            return None

    async def update(self, record_id: UUID, updates: dict[str, Any]) -> bool:
        """
        Partially update a record.

        Args:
            record_id: UUID of record to update
            updates: Dict of fields to update

        Returns:
            True if updated, False if not found
        """
        try:
            updates["updated_at"] = _utc_now().isoformat()
            await self._client.update(
                index=self.index_name,
                id=str(record_id),
                doc=updates,
            )
            logger.debug(f"Updated record {record_id} in {self.index_name}")
            return True
        except NotFoundError:
            return False

    async def delete(self, record_id: UUID) -> bool:
        """Delete a record by UUID."""
        try:
            await self._client.delete(
                index=self.index_name,
                id=str(record_id),
            )
            logger.debug(f"Deleted record {record_id} from {self.index_name}")
            return True
        except NotFoundError:
            return False

    async def search(
        self,
        query: dict[str, Any],
        size: int = 10,
    ) -> list[T]:
        """
        Search for records.

        Args:
            query: Elasticsearch query DSL
            size: Max results to return

        Returns:
            List of matching records as Pydantic models
        """
        response = await self._client.search(
            index=self.index_name,
            query=query,
            size=size,
        )

        return [
            self.record_class.model_validate(hit["_source"])
            for hit in response["hits"]["hits"]
        ]

    async def exists(self, record_id: UUID) -> bool:
        """Check if a record exists."""
        return await self._client.exists(index=self.index_name, id=str(record_id))


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

        Creates a WhoIWasRecord before updating to preserve history.
        """
        # Get current record before updating
        current = await self.get(record_id)
        if current is None:
            return False

        # Create WhoIWasRecord to preserve history
        who_i_was = WhoIWasRecord(
            supersedes=record_id,
            reason=updates.get("_change_reason", "Updated via API"),
            source_type=current.source_type,
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


class MainStore(BaseElasticsearchStore):
    """Store for all relevant content - originals and compressions."""

    index_name = "store"
    record_class = StoreRecord


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
    ) -> UUID:
        """
        Archive a record to the forgotten store.

        Args:
            record: The record being forgotten
            reason: Why it's being forgotten
            original_store: Which store it came from ('coherence', 'store')

        Returns:
            UUID of the forgotten record
        """
        forgotten = ForgottenRecord(
            source_type=record.source_type,
            zotero_key=record.zotero_key,
            forgotten_reason=reason,
            original_store=original_store,
        )

        await self.add(forgotten)
        logger.info(f"Archived record to forgotten: {forgotten.id}, reason: {reason}")
        return forgotten.id
