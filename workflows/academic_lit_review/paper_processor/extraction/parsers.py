"""Output parsing utilities for content fetching."""

import logging
from uuid import UUID, uuid4

from langchain_tools.base import get_store_manager
from core.stores.schema import StoreRecord, SourceType
from workflows.shared.text_utils import count_words

from .prompts import L0_SIZE_THRESHOLD_FOR_L2

logger = logging.getLogger(__name__)


async def _generate_l2_from_l0(
    store_manager,
    l0_record_id: UUID,
    l0_content: str,
    zotero_key: str | None = None,
) -> str | None:
    """Generate L2 (10:1 summary) from L0 content when L2 doesn't exist.

    This is a fallback for cached papers that have L0 but no L2.
    Uses chapter detection and summarization from document_processing.

    Args:
        store_manager: Store manager instance
        l0_record_id: UUID of the L0 record
        l0_content: The L0 markdown content
        zotero_key: Optional Zotero key for the record

    Returns:
        L2 content if successfully generated, None otherwise
    """
    from workflows.document_processing.nodes.chapter_detector import (
        detect_chapters,
        _create_fallback_chunks,
    )
    from workflows.document_processing.subgraphs.chapter_summarization.chunking import (
        chunk_large_content,
    )
    from workflows.document_processing.subgraphs.chapter_summarization.nodes import (
        _summarize_content_chunk,
    )

    try:
        word_count = count_words(l0_content)
        logger.info(
            f"Generating L2 from L0 ({len(l0_content)} chars, {word_count} words) "
            f"for record {l0_record_id}"
        )

        # Skip if document is too short (same threshold as document_processing)
        if word_count < 3000:
            logger.info(f"Document too short ({word_count} words), skipping L2 generation")
            return None

        # Create fallback chunks (simpler than full chapter detection for papers)
        # Papers typically don't have chapter structure, so use size-based chunking
        chunks = _create_fallback_chunks(l0_content, word_count)

        if not chunks:
            logger.warning("No chunks created, cannot generate L2")
            return None

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_content = l0_content[chunk.start_position:chunk.end_position]
            target_words = max(50, chunk.word_count // 10)  # 10:1 compression

            # Handle very large chunks by sub-chunking
            sub_chunks = chunk_large_content(chunk_content)

            if len(sub_chunks) == 1:
                summary = await _summarize_content_chunk(
                    content=chunk_content,
                    target_words=target_words,
                    chapter_context=f"Section {i + 1} of {len(chunks)}",
                )
            else:
                # Large chunk - summarize sub-chunks
                sub_target = max(50, target_words // len(sub_chunks))
                sub_summaries = []
                for j, sub_chunk in enumerate(sub_chunks):
                    sub_summary = await _summarize_content_chunk(
                        content=sub_chunk,
                        target_words=sub_target,
                        chapter_context=f"Section {i + 1} of {len(chunks)}",
                        chunk_num=j + 1,
                        total_chunks=len(sub_chunks),
                    )
                    sub_summaries.append(sub_summary)
                summary = "\n\n".join(sub_summaries)

            chunk_summaries.append(summary)

        # Combine summaries
        tenth_summary = "\n\n".join(chunk_summaries)
        logger.info(
            f"Generated L2 summary: {count_words(tenth_summary)} words "
            f"(from {word_count} words, {len(chunks)} chunks)"
        )

        # Save L2 to store
        l2_record_id = uuid4()
        embedding = await store_manager.embedding.embed_long(tenth_summary)

        l2_record = StoreRecord(
            id=l2_record_id,
            source_type=SourceType.INTERNAL,
            zotero_key=zotero_key,
            content=tenth_summary,
            compression_level=2,
            source_ids=[l0_record_id],
            metadata={
                "type": "tenth_summary",
                "word_count": count_words(tenth_summary),
                "generated_from": "extraction_fallback",
            },
            embedding=embedding,
            embedding_model=store_manager.embedding.model,
        )

        await store_manager.es_stores.store.add(l2_record)
        logger.info(f"Saved L2 record {l2_record_id} (source: {l0_record_id})")

        return tenth_summary

    except Exception as e:
        logger.error(f"Failed to generate L2 from L0: {e}")
        return None


async def _fetch_content_for_extraction(store_manager, es_record_id: str, doi: str) -> str | None:
    """Fetch content for extraction, preferring L2 (10:1 summary) over L0 (original).

    L2 is preferred because:
    - For books/long documents, L2 captures the entire content in compressed form
    - L0 would require truncation for long documents, losing most of the content
    - L2 fits comfortably in LLM context windows

    Falls back to L0 if L2 is not available (e.g., short papers that skip 10:1 summarization).
    If L0 is too large (>150k chars), generates L2 on-the-fly and saves it for future use.

    Note: es_record_id is the L0 UUID. For L2, we use get_by_source_id() which
    searches by source_ids field (L1/L2 records store L0 UUID in source_ids).
    """
    record_uuid = UUID(es_record_id)
    store = store_manager.es_stores.store

    # Try L2 first (10:1 summary) - better for long documents
    # Use get_by_source_id since es_record_id is the L0 UUID
    record = await store.get_by_source_id(record_uuid, compression_level=2)
    if record and record.content:
        logger.debug(f"Using L2 (10:1 summary) for {doi}")
        return record.content

    # Fall back to L0 (original) for papers without L2
    l0_record = await store.get(record_uuid, compression_level=0)
    if not l0_record or not l0_record.content:
        return None

    l0_content = l0_record.content

    # If L0 is too large, generate L2 on-the-fly
    if len(l0_content) > L0_SIZE_THRESHOLD_FOR_L2:
        logger.info(
            f"L0 content for {doi} is {len(l0_content)} chars (>{L0_SIZE_THRESHOLD_FOR_L2}), "
            f"generating L2 summary"
        )
        l2_content = await _generate_l2_from_l0(
            store_manager=store_manager,
            l0_record_id=record_uuid,
            l0_content=l0_content,
            zotero_key=l0_record.zotero_key,
        )
        if l2_content:
            return l2_content
        # If L2 generation failed, fall through to return L0
        logger.warning(f"L2 generation failed for {doi}, falling back to L0")

    logger.debug(f"Using L0 (original) for {doi}")
    return l0_content
