"""
Update the initial store record with processed content.
"""

from uuid import UUID

from langchain_tools.base import get_store_manager


async def update_store(state: dict) -> dict:
    """
    Update the initial store record with processed content.

    Updates:
    - StoreRecord content with full markdown
    - Generates embedding
    - Adds chunks to Chroma for vector search
    """
    store_manager = get_store_manager()
    processing_result = state["processing_result"]
    store_records = state["store_records"]

    if not store_records:
        raise ValueError("No store records found to update")

    # Get the initial record ID
    record_id = UUID(store_records[0]["id"])

    # Update Elasticsearch record
    await store_manager.es_stores.store.update(
        record_id,
        {
            "content": processing_result["markdown"],
            "metadata": {
                "word_count": processing_result["word_count"],
                "page_count": processing_result["page_count"],
                "ocr_method": processing_result["ocr_method"],
                "processing_status": "completed",
            },
        },
    )

    # Get the updated record for embedding
    updated_record = await store_manager.es_stores.store.get(record_id)
    if not updated_record:
        raise ValueError(f"Record {record_id} not found after update")

    # Generate embedding for full content
    embedding = await store_manager.embedding.embed(processing_result["markdown"])

    # Add to Chroma for vector search
    await store_manager.chroma.add(
        record=updated_record,
        embedding=embedding,
        document=processing_result["markdown"],
    )

    # Also add chunks to Chroma for granular search
    for chunk in processing_result.get("chunks", []):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        chunk_embedding = await store_manager.embedding.embed(chunk_text)

        # Create a pseudo-record for the chunk (shares parent metadata)
        from core.stores.schema import StoreRecord, SourceType
        from uuid import uuid4

        chunk_record = StoreRecord(
            id=uuid4(),
            source_type=SourceType.INTERNAL,
            content=chunk_text,
            compression_level=0,
            source_ids=[record_id],  # Link to parent
            metadata={
                "parent_id": str(record_id),
                "heading": chunk.get("heading"),
                "level": chunk.get("level"),
                "chunk_type": "heading_section",
            },
        )

        await store_manager.chroma.add(
            record=chunk_record,
            embedding=chunk_embedding,
            document=chunk_text,
        )

    # Update store records with content preview
    updated_store_records = [{
        "id": str(record_id),
        "compression_level": 0,
        "content_preview": processing_result["markdown"][:200],
    }]

    return {
        "store_records": updated_store_records,
        "current_status": "store_updated",
    }
