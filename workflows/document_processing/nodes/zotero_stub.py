"""
Create Zotero item and initial store record as tracking stubs.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core.stores.schema import SourceType, StoreRecord
from core.stores.zotero import ZoteroItemCreate, ZoteroTag
from langchain_tools.base import get_store_manager


async def create_zotero_stub(state: dict) -> dict:
    """
    Create Zotero item and initial store record as tracking stubs.

    Creates:
    - Zotero item with available metadata + "pending" tag
    - Placeholder StoreRecord (compression_level=0)
    """
    input_data = state["input"]
    store_manager = get_store_manager()

    # Derive title from source if not provided
    title = input_data.get("title")
    if not title:
        source = input_data["source"]
        if state.get("source_type") == "url":
            title = Path(source).stem or source
        elif state.get("source_type") == "local_file":
            title = Path(source).stem
        else:
            title = "Untitled Document"

    # Create Zotero item
    tags = [ZoteroTag(tag=tag) for tag in input_data.get("extra_metadata", {}).get("tags", [])]
    tags.append(ZoteroTag(tag="pending", type=1))  # Auto-tag

    zotero_item = ZoteroItemCreate(
        itemType=input_data.get("item_type", "document"),
        fields={
            "title": title,
            **input_data.get("extra_metadata", {}),
        },
        tags=tags,
    )

    zotero_key = await store_manager.zotero.add(zotero_item)

    # Create placeholder store record
    record_id = uuid4()
    store_record = StoreRecord(
        id=record_id,
        source_type=SourceType.EXTERNAL,
        zotero_key=zotero_key,
        content="",  # Will be updated after processing
        compression_level=0,
        metadata={
            "title": title,
            "processing_status": "pending",
            "source": input_data["source"],
            "doi": input_data.get("extra_metadata", {}).get("DOI"),
        },
    )

    await store_manager.es_stores.store.add(store_record)

    return {
        "zotero_key": zotero_key,
        "store_records": [{
            "id": str(record_id),
            "compression_level": 0,
            "content_preview": "",
        }],
        "started_at": datetime.now(timezone.utc),
        "current_status": "stub_created",
    }
