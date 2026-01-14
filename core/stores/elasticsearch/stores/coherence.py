"""CoherenceStore for identity, beliefs, preferences with confidence scores."""

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from elasticsearch import AsyncElasticsearch, NotFoundError

from ...schema import CoherenceRecord, WhoIWasRecord, _utc_now
from ..base import BaseElasticsearchStore

if TYPE_CHECKING:
    from ..client import ElasticsearchStores

logger = logging.getLogger(__name__)


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
            f"Updated coherence record {record_id}, version saved as {who_i_was.id}"
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
