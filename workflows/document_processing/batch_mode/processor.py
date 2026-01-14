"""Main batch processing logic for document processing."""

import logging
from typing import Any, Optional

from workflows.shared.batch_processor import BatchProcessor, BatchResult

from .job_manager import JobManager
from .types import BatchDocumentResult

logger = logging.getLogger(__name__)


class BatchDocumentProcessor:
    """
    Process multiple documents using Anthropic's Message Batches API.

    Provides 50% cost reduction by batching all LLM calls together.

    Example:
        processor = BatchDocumentProcessor()

        # Add documents
        processor.add_document(
            document_id="doc1",
            content="Full document text...",
            title="My Document",
        )
        processor.add_document(
            document_id="doc2",
            content="Another document...",
        )

        # Execute batch (may take up to 1 hour)
        results = await processor.execute()
        print(results["doc1"].summary)
    """

    def __init__(self, poll_interval: int = 60):
        """
        Initialize batch document processor.

        Args:
            poll_interval: Seconds between batch status checks (default: 60)
        """
        self.batch_processor = BatchProcessor(poll_interval=poll_interval)
        self.job_manager = JobManager(self.batch_processor)

    def add_document(
        self,
        document_id: str,
        content: str,
        title: Optional[str] = None,
        summary_target_words: int = 100,
        include_metadata: bool = True,
        include_chapter_summaries: bool = False,
        chapters: Optional[list[dict]] = None,
    ) -> None:
        """
        Add a document for batch processing.

        Args:
            document_id: Unique identifier for this document
            content: Full document text (markdown)
            title: Optional document title
            summary_target_words: Target word count for summary (default: 100)
            include_metadata: Extract metadata (default: True)
            include_chapter_summaries: Generate chapter summaries (default: False)
            chapters: Chapter info if include_chapter_summaries is True
        """
        self.job_manager.add_document(
            document_id=document_id,
            content=content,
            title=title,
            summary_target_words=summary_target_words,
            include_metadata=include_metadata,
            include_chapter_summaries=include_chapter_summaries,
            chapters=chapters,
        )

    def _parse_metadata(self, content: str) -> dict:
        """Parse JSON metadata from response."""
        import json

        content = content.strip()

        # Extract from markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def _process_results(
        self,
        batch_results: dict[str, BatchResult],
    ) -> dict[str, BatchDocumentResult]:
        """Process batch results into document results."""
        results: dict[str, BatchDocumentResult] = {}

        # Initialize results for all pending documents
        for doc_id in self.job_manager.pending_documents:
            results[doc_id] = BatchDocumentResult(document_id=doc_id)

        # Process each batch result
        for custom_id, result in batch_results.items():
            parts = custom_id.split(":")
            doc_id = parts[0]
            result_type = parts[1]

            if doc_id not in results:
                continue

            doc_result = results[doc_id]

            if not result.success:
                doc_result.errors.append(f"{result_type}: {result.error}")
                continue

            if result_type == "summary":
                doc_result.summary = result.content

            elif result_type == "metadata":
                doc_result.metadata = self._parse_metadata(result.content)

            elif result_type == "chapter":
                chapter_idx = int(parts[2])
                doc = self.job_manager.pending_documents[doc_id]
                chapter_info = doc.chapters[chapter_idx] if doc.chapters else {}

                if doc_result.chapter_summaries is None:
                    doc_result.chapter_summaries = []

                doc_result.chapter_summaries.append(
                    {
                        "title": chapter_info.get(
                            "title", f"Chapter {chapter_idx + 1}"
                        ),
                        "author": chapter_info.get("author"),
                        "summary": result.content,
                    }
                )

        return results

    async def execute(
        self,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, BatchDocumentResult]:
        """
        Execute batch processing for all pending documents.

        This submits all LLM requests to Anthropic's Message Batches API
        and waits for results. Typically completes within 1 hour.

        Args:
            progress_callback: Optional async callback(batch_id, status, counts)
                              called periodically during processing

        Returns:
            Dictionary mapping document_id to BatchDocumentResult

        Raises:
            RuntimeError: If batch processing fails
        """
        if not self.job_manager.pending_documents:
            return {}

        logger.info(
            f"Starting batch processing for {len(self.job_manager.pending_documents)} documents"
        )

        # Queue all requests
        self.job_manager.queue_batch_requests()

        # Execute batch
        if progress_callback:
            batch_results = await self.batch_processor.execute_batch_with_callback(
                callback=progress_callback,
                callback_interval=300,
            )
        else:
            batch_results = await self.batch_processor.execute_batch()

        # Process results
        results = self._process_results(batch_results)

        # Clear pending documents
        self.job_manager.clear()

        succeeded = sum(1 for r in results.values() if not r.errors)
        logger.info(f"Batch processing complete: {succeeded}/{len(results)} succeeded")

        return results


async def process_documents_with_batch_api(
    documents: list[dict[str, Any]],
    include_metadata: bool = True,
    include_chapter_summaries: bool = False,
    progress_callback: Optional[callable] = None,
) -> dict[str, BatchDocumentResult]:
    """
    Process multiple documents using Anthropic's Message Batches API.

    This is the recommended approach for bulk processing when:
    - You have multiple documents to process
    - Immediate results aren't required
    - Cost savings (50%) are important

    Args:
        documents: List of document dicts with:
            - id: Unique document identifier
            - content: Full document text (markdown)
            - title: Optional document title
            - chapters: Optional list of chapter info dicts
        include_metadata: Extract metadata (default: True)
        include_chapter_summaries: Generate chapter summaries (default: False)
        progress_callback: Optional async callback for progress updates

    Returns:
        Dictionary mapping document id to BatchDocumentResult

    Example:
        results = await process_documents_with_batch_api([
            {"id": "doc1", "content": "Document 1 text...", "title": "Doc 1"},
            {"id": "doc2", "content": "Document 2 text...", "title": "Doc 2"},
        ])

        for doc_id, result in results.items():
            print(f"{doc_id}: {result.summary[:100]}...")
    """
    processor = BatchDocumentProcessor()

    for doc in documents:
        processor.add_document(
            document_id=doc["id"],
            content=doc["content"],
            title=doc.get("title"),
            include_metadata=include_metadata,
            include_chapter_summaries=include_chapter_summaries,
            chapters=doc.get("chapters"),
        )

    return await processor.execute(progress_callback=progress_callback)
