"""
Update Zotero item with extracted data node.
"""

import logging
from typing import Any

from core.stores.zotero import ZoteroCreator, ZoteroItemUpdate
from langchain_tools.base import get_store_manager
from workflows.document_processing.state import DocumentProcessingState

logger = logging.getLogger(__name__)


async def update_zotero(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Update Zotero item with summary and extracted metadata.

    Sets abstractNote to short_summary, updates authors/date/publisher
    from metadata_updates, removes "pending" tag and adds "processed" tag.

    Returns current_status.
    """
    try:
        zotero_key = state.get("zotero_key")
        if not zotero_key:
            logger.warning("No zotero_key in state, skipping Zotero update")
            return {"current_status": "update_zotero_skipped"}

        short_summary = state.get("short_summary")
        metadata_updates = state.get("metadata_updates", {})

        store_manager = get_store_manager()

        # Build update payload
        update = ZoteroItemUpdate()

        # Update fields
        fields = {}
        if short_summary:
            fields["abstractNote"] = short_summary
        if "title" in metadata_updates:
            fields["title"] = metadata_updates["title"]
        if "date" in metadata_updates:
            fields["date"] = metadata_updates["date"]
        if "publisher" in metadata_updates:
            fields["publisher"] = metadata_updates["publisher"]
        if "isbn" in metadata_updates:
            fields["ISBN"] = metadata_updates["isbn"]

        if fields:
            update.fields = fields

        # Update authors if present
        if "authors" in metadata_updates and metadata_updates["authors"]:
            creators = []
            for author_name in metadata_updates["authors"]:
                # Parse name (simple approach: assume "First Last" format)
                parts = author_name.strip().split(" ", 1)
                if len(parts) == 2:
                    creators.append(
                        ZoteroCreator(
                            firstName=parts[0], lastName=parts[1], creatorType="author"
                        )
                    )
                else:
                    # Single name or complex format
                    creators.append(
                        ZoteroCreator(name=author_name, creatorType="author")
                    )
            update.creators = creators

        # Update tags: remove "pending", add "processed"
        # First, get current item to preserve other tags
        current_item = await store_manager.zotero.get(zotero_key)
        if current_item:
            current_tags = [
                tag.get("tag", "") for tag in current_item.tags if isinstance(tag, dict)
            ]
            # Remove "pending", add "processed"
            new_tags = [t for t in current_tags if t.lower() != "pending"]
            if "processed" not in [t.lower() for t in new_tags]:
                new_tags.append("processed")
            update.tags = new_tags
        else:
            update.tags = ["processed"]

        # Perform update
        success = await store_manager.zotero.update(zotero_key, update)

        if success:
            logger.info(f"Updated Zotero item: {zotero_key}")
            return {"current_status": "update_zotero_complete"}
        else:
            logger.warning(f"Zotero item not found: {zotero_key}")
            return {"current_status": "update_zotero_not_found"}

    except Exception as e:
        logger.error(f"Failed to update Zotero: {e}")
        return {
            "current_status": "update_zotero_failed",
            "errors": [{"node": "update_zotero", "error": str(e)}],
        }
