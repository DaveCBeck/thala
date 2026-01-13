"""ForgottenStore for archived/forgotten content."""

import logging
from typing import Optional
from uuid import UUID

from elasticsearch import AsyncElasticsearch

from ...schema import BaseRecord, ForgottenRecord
from ..base import BaseElasticsearchStore

logger = logging.getLogger(__name__)


class ForgottenStore(BaseElasticsearchStore):
    """Store for archived/forgotten content."""

    index_name = "forgotten"
    record_class = ForgottenRecord

    def __init__(self, client: AsyncElasticsearch):
        super().__init__(client)

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
        logger.debug(f"Archived record {forgotten.id} to forgotten: {reason}")
        return forgotten.id
