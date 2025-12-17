"""
document_processing - LangChain tool to trigger document processing workflow.

Processes documents through extraction, summarization, and metadata enrichment.
"""

import logging
from typing import Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentProcessingOutput(BaseModel):
    """Output schema for process_document tool."""

    success: bool
    zotero_key: Optional[str] = None
    store_records: list[dict] = Field(default_factory=list)
    short_summary: Optional[str] = None
    tenth_summary: Optional[str] = None
    word_count: int = 0
    status: str = "unknown"
    errors: list[dict] = Field(default_factory=list)


@tool
async def process_document(
    source: str,
    title: Optional[str] = None,
    item_type: Literal[
        "book", "journalArticle", "document", "webpage", "report", "thesis"
    ] = "document",
    quality: Literal["fast", "balanced", "quality"] = "balanced",
    authors: Optional[list[str]] = None,
    date: Optional[str] = None,
) -> dict:
    """Process a document through the full extraction and summarization pipeline.

    Use this when you need to:
    - Add a new book, paper, or article to the knowledge base
    - Process a PDF, DOCX, EPUB, or other document format
    - Create searchable summaries at multiple compression levels
    - Extract and store metadata from documents

    The workflow:
    1. Creates tracking entries in Zotero and store
    2. Extracts text via Marker (or chunks markdown directly)
    3. Generates a 100-word summary
    4. Extracts/verifies metadata
    5. For documents >2000 words, creates a 10:1 chapter-by-chapter summary

    Args:
        source: File path, URL, or markdown text to process
        title: Optional title (auto-derived from source if not provided)
        item_type: Zotero item type for cataloging
        quality: Marker processing quality (fast/balanced/quality)
        authors: Optional list of author names ("Last, First" format)
        date: Optional publication date (YYYY or YYYY-MM-DD format)
    """
    try:
        from workflows.document_processing import process_document as run_workflow
    except ImportError as e:
        logger.error(f"Failed to import workflow: {e}")
        return DocumentProcessingOutput(
            success=False,
            status="import_error",
            errors=[{"error": str(e)}],
        ).model_dump(mode="json")

    extra_metadata = {}
    if authors:
        extra_metadata["authors"] = authors
    if date:
        extra_metadata["date"] = date

    try:
        result = await run_workflow(
            source=source,
            title=title,
            item_type=item_type,
            quality=quality,
            langs=["English"],
            extra_metadata=extra_metadata,
        )

        word_count = 0
        if result.get("processing_result"):
            word_count = result["processing_result"].get("word_count", 0)

        output = DocumentProcessingOutput(
            success=result.get("current_status") in ("completed", "finalized"),
            zotero_key=result.get("zotero_key"),
            store_records=result.get("store_records", []),
            short_summary=result.get("short_summary"),
            tenth_summary=result.get("tenth_summary"),
            word_count=word_count,
            status=result.get("current_status", "unknown"),
            errors=result.get("errors", []),
        )

        return output.model_dump(mode="json")

    except Exception as e:
        logger.exception(f"Document processing failed: {e}")
        return DocumentProcessingOutput(
            success=False,
            status="error",
            errors=[{"error": str(e)}],
        ).model_dump(mode="json")
