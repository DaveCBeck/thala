"""
Batch-mode document processing using Anthropic Message Batches API.

This module provides 50% cost reduction by batching all LLM calls
together and submitting them to Anthropic's Message Batches API.

Use this when:
- Processing multiple documents that don't need immediate results
- Cost savings are more important than latency
- You're doing bulk processing or large-scale analysis

Note: Batch processing is asynchronous. Results may take up to 24 hours
(typically completes within 1 hour).
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from workflows.shared.batch_processor import BatchProcessor, BatchResult, ModelTier

logger = logging.getLogger(__name__)


@dataclass
class BatchDocumentRequest:
    """Request for batch document summarization."""
    document_id: str
    content: str
    title: Optional[str] = None
    summary_target_words: int = 100
    include_metadata: bool = True
    include_chapter_summaries: bool = False
    chapters: Optional[list[dict]] = None


@dataclass
class BatchDocumentResult:
    """Result from batch document processing."""
    document_id: str
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    chapter_summaries: Optional[list[dict]] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


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
        self.pending_documents: dict[str, BatchDocumentRequest] = {}

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
        self.pending_documents[document_id] = BatchDocumentRequest(
            document_id=document_id,
            content=content,
            title=title,
            summary_target_words=summary_target_words,
            include_metadata=include_metadata,
            include_chapter_summaries=include_chapter_summaries,
            chapters=chapters,
        )

    def _build_summary_prompt(self, doc: BatchDocumentRequest) -> str:
        """Build prompt for document summarization."""
        prompt = f"Summarize the following text in approximately {doc.summary_target_words} words."
        prompt += "\n\nCreate a concise summary capturing the main thesis, key arguments, and conclusions."
        prompt += " Focus on what makes this work significant and its core contributions."

        # Use first and last portions for long documents
        content = doc.content
        if len(content) > 50000:
            # Approximate first and last 10 pages
            first_part = content[:25000]
            last_part = content[-25000:]
            content = f"{first_part}\n\n[... middle section omitted ...]\n\n{last_part}"

        prompt += f"\n\nText:\n{content}"
        return prompt

    def _build_metadata_prompt(self, doc: BatchDocumentRequest) -> str:
        """Build prompt for metadata extraction."""
        prompt = """Extract bibliographic metadata from this document excerpt. Look for:
- title: Full document title
- authors: List of author names (can be empty list)
- date: Publication date (any format)
- publisher: Publisher name
- isbn: ISBN if present

Also determine:
- is_multi_author: true if this appears to be a multi-author edited volume
- chapter_authors: dict mapping chapter titles to author names (only for multi-author books)

Return ONLY a JSON object with these fields. Use null for missing values, empty list for no authors."""

        # Use first and last portions for metadata
        content = doc.content
        first_part = content[:15000]
        last_part = content[-15000:] if len(content) > 15000 else ""
        excerpt = f"{first_part}\n\n--- END OF FRONT MATTER ---\n\n{last_part}" if last_part else first_part

        prompt += f"\n\nText:\n{excerpt}"
        return prompt

    def _build_chapter_summary_prompt(
        self,
        chapter_title: str,
        chapter_content: str,
        target_words: int,
        chapter_author: Optional[str] = None,
    ) -> str:
        """Build prompt for chapter summarization."""
        context = f"Chapter: {chapter_title}"
        if chapter_author:
            context += f" (by {chapter_author})"

        prompt = f"""Summarize this chapter in approximately {target_words} words.

Context: {context}

Focus on:
- The main arguments and thesis of the chapter
- Key concepts and findings
- How this chapter contributes to the broader work
- Any significant conclusions or implications

Provide a coherent, well-structured summary that captures the essential content.

Text:
{chapter_content}"""
        return prompt

    def _queue_batch_requests(self) -> None:
        """Add all pending documents to the batch processor."""
        for doc_id, doc in self.pending_documents.items():
            # Queue summary request (Sonnet)
            self.batch_processor.add_request(
                custom_id=f"{doc_id}:summary",
                prompt=self._build_summary_prompt(doc),
                model=ModelTier.SONNET,
                max_tokens=1024,
            )

            # Queue metadata request (Sonnet)
            if doc.include_metadata:
                self.batch_processor.add_request(
                    custom_id=f"{doc_id}:metadata",
                    prompt=self._build_metadata_prompt(doc),
                    model=ModelTier.SONNET,
                    max_tokens=2048,
                )

            # Queue chapter summaries (Opus with thinking)
            if doc.include_chapter_summaries and doc.chapters:
                for i, chapter in enumerate(doc.chapters):
                    chapter_content = doc.content[
                        chapter["start_position"]:chapter["end_position"]
                    ]
                    target_words = max(50, chapter.get("word_count", 500) // 10)

                    self.batch_processor.add_request(
                        custom_id=f"{doc_id}:chapter:{i}",
                        prompt=self._build_chapter_summary_prompt(
                            chapter_title=chapter["title"],
                            chapter_content=chapter_content,
                            target_words=target_words,
                            chapter_author=chapter.get("author"),
                        ),
                        # TODO: Upgrade to ModelTier.OPUS before production
                        model=ModelTier.HAIKU,
                        max_tokens=12000,
                        thinking_budget=8000,
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
        for doc_id in self.pending_documents:
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
                doc = self.pending_documents[doc_id]
                chapter_info = doc.chapters[chapter_idx] if doc.chapters else {}

                if doc_result.chapter_summaries is None:
                    doc_result.chapter_summaries = []

                doc_result.chapter_summaries.append({
                    "title": chapter_info.get("title", f"Chapter {chapter_idx + 1}"),
                    "author": chapter_info.get("author"),
                    "summary": result.content,
                })

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
        if not self.pending_documents:
            return {}

        logger.info(f"Starting batch processing for {len(self.pending_documents)} documents")

        # Queue all requests
        self._queue_batch_requests()

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
        self.pending_documents.clear()

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
