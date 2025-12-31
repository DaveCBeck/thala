"""WhoIWasStore for edit history - temporal records of superseded state."""

from uuid import UUID

from elasticsearch import AsyncElasticsearch

from ...schema import WhoIWasRecord
from ..base import BaseElasticsearchStore


class WhoIWasStore(BaseElasticsearchStore):
    """Store for edit history - temporal records of superseded state."""

    index_name = "who_i_was"
    record_class = WhoIWasRecord

    def __init__(self, client: AsyncElasticsearch):
        super().__init__(client)

    async def get_history(self, record_id: UUID) -> list[WhoIWasRecord]:
        """Get all historical versions that superseded this record."""
        return await self.search(
            query={"term": {"supersedes": str(record_id)}},
            size=100,
        )
