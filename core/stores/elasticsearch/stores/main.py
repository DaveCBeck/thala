"""MainStore for all relevant content - originals and compressions."""

import logging
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from elasticsearch import AsyncElasticsearch, NotFoundError

from ...schema import BaseRecord, StoreRecord
from ..base import BaseElasticsearchStore

if TYPE_CHECKING:
    from ..client import ElasticsearchStores

logger = logging.getLogger(__name__)


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

    async def get_by_source_id(
        self,
        source_id: UUID,
        compression_level: int,
    ) -> Optional[StoreRecord]:
        """
        Get a compressed record by its source (L0) UUID.

        L1 and L2 records store the L0 UUID in their source_ids field.
        This method finds the compressed record derived from a given L0 record.

        Args:
            source_id: UUID of the source L0 record
            compression_level: Which compression level to search (1 or 2)

        Returns:
            StoreRecord if found, None otherwise
        """
        if compression_level == 0:
            # For L0, use direct lookup
            return await self.get(source_id, compression_level=0)

        index = self._index_for_level(compression_level)
        query = {
            "bool": {
                "must": [
                    {"term": {"source_ids": str(source_id)}},
                    {"term": {"compression_level": compression_level}},
                ]
            }
        }

        results = await super().search(query, size=1, index=index)
        if results:
            return results[0]
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
