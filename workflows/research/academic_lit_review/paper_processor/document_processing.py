"""Document processing wrapper for paper pipeline."""

import asyncio
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.stores.schema import SourceType, StoreRecord
from core.stores.zotero import ZoteroItemCreate, ZoteroItemUpdate, ZoteroTag
from langchain_tools.base import get_store_manager
from workflows.document_processing.batch_mode import BatchDocumentProcessor
from workflows.document_processing.graph import process_document
from workflows.research.academic_lit_review.state import PaperMetadata

logger = logging.getLogger(__name__)


async def process_single_document(
    doi: str,
    source: str,
    paper: PaperMetadata,
    is_markdown: bool = False,
    use_batch_api: bool = True,
) -> dict[str, Any]:
    """Process a single document through document_processing workflow.

    Args:
        doi: Paper DOI
        source: Local file path OR markdown content (if is_markdown=True)
        paper: Paper metadata
        is_markdown: If True, source is markdown text, not a file path
        use_batch_api: If False, skip batch API for rapid iteration

    Returns:
        Processing result with es_record_id, zotero_key, etc.
    """
    try:
        if is_markdown:
            # Source is markdown content from OA HTML scrape
            logger.info(f"Processing {doi}: markdown content ({len(source)} chars)")
        else:
            # Source is a file path
            source_path = Path(source)
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source}")
                return {
                    "doi": doi,
                    "success": False,
                    "zotero_key": None,
                    "errors": [
                        {
                            "node": "process_document",
                            "error": f"File not found: {source}",
                        }
                    ],
                }

            file_size_mb = source_path.stat().st_size / (1024 * 1024)
            logger.info(f"Processing {doi}: {source} ({file_size_mb:.1f} MB)")

        extra_metadata = {
            "DOI": doi,
            "date": paper.get("publication_date", ""),
            "publicationTitle": paper.get("venue", ""),
            "abstractNote": paper.get("abstract", "")[:500]
            if paper.get("abstract")
            else "",
        }

        result = await process_document(
            source=source,
            title=paper.get("title", "Unknown"),
            item_type="journalArticle",
            extra_metadata=extra_metadata,
            use_batch_api=use_batch_api,
        )

        status = result.get("current_status", "unknown")
        errors = result.get("errors", [])
        logger.info(
            f"Document {doi} processed with status: {status}, errors: {len(errors)}"
        )
        if errors:
            for err in errors:
                logger.warning(
                    f"  {err.get('node', 'unknown')}: {err.get('error', 'no details')}"
                )

        # Extract ES record ID from store_records list (first record at compression level 0)
        store_records = result.get("store_records", [])
        es_record_id = store_records[0].get("id") if store_records else None

        return {
            "doi": doi,
            "success": status not in ("failed",),
            "es_record_id": es_record_id,
            "zotero_key": result.get("zotero_key"),
            "short_summary": result.get("short_summary", ""),
            "original_language": result.get("original_language", "en"),
            "errors": errors,
        }

    except TimeoutError as e:
        logger.error(f"Timeout processing document {doi}: {e}")
        return {
            "doi": doi,
            "success": False,
            "zotero_key": None,
            "errors": [{"node": "process_document", "error": f"Timeout: {e}"}],
        }
    except Exception as e:
        logger.error(f"Failed to process document {doi}: {type(e).__name__}: {e}")
        return {
            "doi": doi,
            "success": False,
            "zotero_key": None,
            "errors": [
                {"node": "process_document", "error": f"{type(e).__name__}: {e}"}
            ],
        }


async def process_multiple_documents(
    documents: list[tuple[str, str, PaperMetadata]],
    use_batch_api: bool = True,
) -> list[dict[str, Any]]:
    """Process multiple documents with batched LLM calls.

    Uses BatchDocumentProcessor to batch all LLM calls (summary, metadata)
    into a single batch API request, then runs store operations per-document.

    Args:
        documents: List of (doi, markdown_text, paper_metadata) tuples
        use_batch_api: If False, fall back to individual processing

    Returns:
        List of processing results in same format as process_single_document
    """
    if not documents:
        return []

    # Fall back to individual processing if batch API disabled
    if not use_batch_api:
        results = []
        for doi, markdown_text, paper in documents:
            result = await process_single_document(
                doi, markdown_text, paper, is_markdown=True, use_batch_api=False
            )
            results.append(result)
        return results

    logger.info(f"Batch processing {len(documents)} documents")
    store_manager = get_store_manager()

    # Track per-document state
    doc_state: dict[str, dict] = {}
    results: list[dict[str, Any]] = []

    # Step 1: Create Zotero stubs + ES records concurrently
    async def create_stub(doi: str, markdown_text: str, paper: PaperMetadata) -> dict:
        """Create Zotero item and initial ES record."""
        try:
            title = paper.get("title", "Unknown")
            extra_metadata = {
                "DOI": doi,
                "date": paper.get("publication_date", ""),
                "publicationTitle": paper.get("venue", ""),
                "abstractNote": paper.get("abstract", "")[:500] if paper.get("abstract") else "",
            }

            # Create Zotero item
            tags = [ZoteroTag(tag="pending", type=1)]
            zotero_item = ZoteroItemCreate(
                itemType="journalArticle",
                fields={"title": title, **extra_metadata},
                tags=tags,
            )
            zotero_key = await store_manager.zotero.add(zotero_item)

            # Create ES stub record
            record_id = uuid4()
            store_record = StoreRecord(
                id=record_id,
                source_type=SourceType.EXTERNAL,
                zotero_key=zotero_key,
                content="",
                compression_level=0,
                metadata={
                    "title": title,
                    "processing_status": "pending",
                    "doi": doi,
                },
            )
            await store_manager.es_stores.store.add(store_record)

            return {
                "doi": doi,
                "zotero_key": zotero_key,
                "record_id": record_id,
                "markdown": markdown_text,
                "paper": paper,
                "title": title,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Failed to create stub for {doi}: {e}")
            return {"doi": doi, "success": False, "error": str(e)}

    stub_tasks = [create_stub(doi, md, paper) for doi, md, paper in documents]
    stub_results = await asyncio.gather(*stub_tasks, return_exceptions=True)

    for result in stub_results:
        if isinstance(result, Exception):
            logger.error(f"Stub creation exception: {result}")
            continue
        if result.get("success"):
            doc_state[result["doi"]] = result
        else:
            # Add failed result immediately
            results.append({
                "doi": result["doi"],
                "success": False,
                "zotero_key": None,
                "es_record_id": None,
                "short_summary": "",
                "original_language": "en",
                "errors": [{"node": "create_stub", "error": result.get("error", "Unknown")}],
            })

    if not doc_state:
        logger.warning("No documents successfully stubbed")
        return results

    # Step 2: Update ES with markdown content concurrently
    async def update_es_content(doi: str) -> bool:
        """Update ES record with markdown content."""
        try:
            state = doc_state[doi]
            word_count = len(state["markdown"].split())
            await store_manager.es_stores.store.update(
                state["record_id"],
                {
                    "content": state["markdown"],
                    "metadata": {
                        "word_count": word_count,
                        "processing_status": "content_stored",
                    },
                },
                compression_level=0,
            )
            state["word_count"] = word_count
            return True
        except Exception as e:
            logger.error(f"Failed to update ES for {doi}: {e}")
            return False

    update_tasks = [update_es_content(doi) for doi in doc_state.keys()]
    await asyncio.gather(*update_tasks, return_exceptions=True)

    # Step 3: Batch LLM calls via BatchDocumentProcessor
    processor = BatchDocumentProcessor()
    for doi, state in doc_state.items():
        processor.add_document(
            document_id=doi,
            content=state["markdown"],
            title=state["title"],
            include_metadata=True,
            include_chapter_summaries=False,
        )

    logger.info(f"Executing batch LLM processing for {len(doc_state)} documents")
    batch_results = await processor.execute()
    logger.info("Batch LLM processing complete")

    # Step 4: Save summaries to ES with embeddings concurrently
    async def save_summary(doi: str) -> bool:
        """Save summary to ES with embedding."""
        try:
            state = doc_state[doi]
            batch_result = batch_results.get(doi)
            if not batch_result or not batch_result.summary:
                logger.warning(f"No summary for {doi}")
                return False

            state["short_summary"] = batch_result.summary
            state["metadata_updates"] = batch_result.metadata or {}

            # Create L1 record with summary
            summary_record_id = uuid4()
            summary_record = StoreRecord(
                id=summary_record_id,
                source_type=SourceType.INTERNAL,
                zotero_key=state["zotero_key"],
                content=batch_result.summary,
                compression_level=1,
                source_ids=[state["record_id"]],
                language_code="en",
                metadata={
                    "type": "short_summary",
                    "word_count": len(batch_result.summary.split()),
                },
            )

            # Generate embedding
            embedding = await store_manager.embedding.embed(batch_result.summary)
            summary_record.embedding = embedding
            summary_record.embedding_model = store_manager.embedding.model
            await store_manager.es_stores.store.add(summary_record)

            return True
        except Exception as e:
            logger.error(f"Failed to save summary for {doi}: {e}")
            return False

    summary_tasks = [save_summary(doi) for doi in doc_state.keys()]
    await asyncio.gather(*summary_tasks, return_exceptions=True)

    # Step 5: Update Zotero with summaries concurrently
    async def update_zotero(doi: str) -> bool:
        """Update Zotero item with summary and metadata."""
        try:
            state = doc_state[doi]
            short_summary = state.get("short_summary", "")

            update = ZoteroItemUpdate()
            fields = {}
            if short_summary:
                fields["abstractNote"] = short_summary

            metadata_updates = state.get("metadata_updates", {})
            if "title" in metadata_updates:
                fields["title"] = metadata_updates["title"]
            if "date" in metadata_updates:
                fields["date"] = metadata_updates["date"]

            if fields:
                update.fields = fields

            # Update tags: remove pending, add processed
            update.tags = ["processed"]

            await store_manager.zotero.update(state["zotero_key"], update)
            return True
        except Exception as e:
            logger.error(f"Failed to update Zotero for {doi}: {e}")
            return False

    zotero_tasks = [update_zotero(doi) for doi in doc_state.keys()]
    await asyncio.gather(*zotero_tasks, return_exceptions=True)

    # Step 6: Build final results
    for doi, state in doc_state.items():
        batch_result = batch_results.get(doi)
        errors = []
        if batch_result and batch_result.errors:
            errors = [{"node": "batch_llm", "error": e} for e in batch_result.errors]

        results.append({
            "doi": doi,
            "success": bool(state.get("short_summary")),
            "es_record_id": str(state["record_id"]),
            "zotero_key": state["zotero_key"],
            "short_summary": state.get("short_summary", ""),
            "original_language": "en",
            "errors": errors,
        })

    logger.info(
        f"Batch processing complete: {sum(1 for r in results if r['success'])}/{len(results)} succeeded"
    )
    return results
