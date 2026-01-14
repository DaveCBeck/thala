"""Job queue management for batch document processing."""

from typing import Optional

from workflows.shared.batch_processor import BatchProcessor
from workflows.shared.llm_utils import ModelTier

from .types import BatchDocumentRequest


class JobManager:
    """Manages document jobs and batch request queuing."""

    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor
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
        """Add a document for batch processing."""
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
        prompt += (
            " Focus on what makes this work significant and its core contributions."
        )

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
        excerpt = (
            f"{first_part}\n\n--- END OF FRONT MATTER ---\n\n{last_part}"
            if last_part
            else first_part
        )

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

    def queue_batch_requests(self) -> None:
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
                        chapter["start_position"] : chapter["end_position"]
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

    def clear(self) -> None:
        """Clear all pending documents."""
        self.pending_documents.clear()
