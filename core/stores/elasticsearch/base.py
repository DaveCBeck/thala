"""Base class for Elasticsearch-backed stores."""

import logging
from typing import Any, Optional, TypeVar
from uuid import UUID

from elasticsearch import AsyncElasticsearch, NotFoundError

from ..schema import BaseRecord, _utc_now

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseRecord)


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

    async def update(
        self, record_id: UUID, updates: dict[str, Any], index: Optional[str] = None
    ) -> bool:
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
