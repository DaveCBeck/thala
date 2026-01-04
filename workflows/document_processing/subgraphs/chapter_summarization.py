"""
Chapter summarization subgraph using map-reduce pattern.

Uses Opus with extended thinking for complex chapter analysis.
Uses Anthropic Batch API for 50% cost reduction when processing 5+ chapters.
Chunks very long chapters (>600k chars) to avoid token limit errors.
"""

import asyncio
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.batch_processor import BatchProcessor
from workflows.shared.text_utils import count_words

logger = logging.getLogger(__name__)

# Maximum content size before chunking (600k chars ≈ 150k tokens, safe for 200k context)
MAX_CHAPTER_CHARS = 600_000
# Target chunk size when splitting large chapters (in characters)
CHUNK_SIZE_CHARS = 500_000
# Overlap between chunks for context continuity
CHUNK_OVERLAP_CHARS = 2000

def _chunk_large_content(content: str) -> list[str]:
    """
    Split large content into chunks that fit within token limits.

    Uses paragraph boundaries when possible, with overlap for context continuity.
    """
    if len(content) <= MAX_CHAPTER_CHARS:
        return [content]

    chunks = []
    current_pos = 0

    while current_pos < len(content):
        # Calculate end position for this chunk
        end_pos = min(current_pos + CHUNK_SIZE_CHARS, len(content))

        if end_pos < len(content):
            # Try to find a paragraph break near the target
            search_start = max(current_pos, end_pos - 5000)
            search_region = content[search_start:end_pos]
            para_break = search_region.rfind("\n\n")

            if para_break != -1:
                end_pos = search_start + para_break + 2
            else:
                # Fall back to word boundary
                while end_pos > current_pos and not content[end_pos].isspace():
                    end_pos -= 1

        chunks.append(content[current_pos:end_pos])

        # Move position, accounting for overlap
        if end_pos < len(content):
            current_pos = max(current_pos + 1, end_pos - CHUNK_OVERLAP_CHARS)
            # Adjust to word boundary
            while current_pos < len(content) and not content[current_pos].isspace():
                current_pos += 1
        else:
            break

    logger.info(f"Split large chapter into {len(chunks)} chunks")
    return chunks


# Static system prompt (cached) - ~200 tokens, saves 90% on cache hits
# Opus at $15/MTok base means cache hits at $1.50/MTok = 90% savings
CHAPTER_SUMMARIZATION_SYSTEM = """You are an expert summarizer specializing in condensing academic and technical content while preserving essential meaning.

Your task is to create a summary that captures:
- The main arguments and thesis of the chapter
- Key concepts and findings
- How this chapter contributes to the broader work
- Any significant conclusions or implications

Provide a coherent, well-structured summary in clear prose. Maintain academic rigor while being accessible."""

MAX_CONCURRENT_CHAPTER_SUMMARIES = 4


async def _summarize_content_chunk(
    content: str,
    target_words: int,
    chapter_context: str,
    chunk_num: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """Summarize a single chunk of content."""
    chunk_info = ""
    if chunk_num is not None and total_chunks is not None:
        chunk_info = f" (Part {chunk_num}/{total_chunks})"

    user_prompt = f"""Summarize this content in approximately {target_words} words.

Context: {chapter_context}{chunk_info}

Content:
{content}"""

    llm = get_llm(
        tier=ModelTier.OPUS,
        thinking_budget=8000,
        max_tokens=8000 + 4096,
    )

    response = await invoke_with_cache(
        llm,
        system_prompt=CHAPTER_SUMMARIZATION_SYSTEM,
        user_prompt=user_prompt,
    )

    # Extract text content from response
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
            elif hasattr(block, "type") and block.type == "text":
                return getattr(block, "text", "")
        return ""
    return response.content


async def _summarize_single_chapter(
    chapter: dict,
    chapter_content: str,
    target_words: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Summarize a single chapter to ~10% of original length with concurrency control.

    Uses Opus with extended thinking for deep analysis of chapter content,
    including title and author context. Implements prompt caching for cost reduction.

    For very long chapters (>600k chars), chunks the content and summarizes each
    chunk separately, then combines the summaries.

    Returns dict with title, author, and summary.
    """
    async with semaphore:
        try:
            # Build context for the chapter
            chapter_context = f"Chapter: {chapter['title']}"
            if chapter.get("author"):
                chapter_context += f" (by {chapter['author']})"

            # Check if content needs chunking
            chunks = _chunk_large_content(chapter_content)

            if len(chunks) == 1:
                # Normal path - single chunk
                summary = await _summarize_content_chunk(
                    content=chapter_content,
                    target_words=target_words,
                    chapter_context=chapter_context,
                )
            else:
                # Large chapter - summarize each chunk then combine
                logger.info(
                    f"Chapter '{chapter['title']}' is too large ({len(chapter_content)} chars), "
                    f"splitting into {len(chunks)} chunks"
                )
                chunk_target_words = max(50, target_words // len(chunks))
                chunk_summaries = []

                for i, chunk in enumerate(chunks, 1):
                    chunk_summary = await _summarize_content_chunk(
                        content=chunk,
                        target_words=chunk_target_words,
                        chapter_context=chapter_context,
                        chunk_num=i,
                        total_chunks=len(chunks),
                    )
                    chunk_summaries.append(chunk_summary)

                # Combine chunk summaries
                summary = "\n\n".join(chunk_summaries)

            logger.info(f"Summarized chapter '{chapter['title']}' to {len(summary.split())} words")

            return {
                "title": chapter["title"],
                "author": chapter.get("author"),
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Failed to summarize chapter '{chapter['title']}': {e}")
            return {
                "title": chapter["title"],
                "author": chapter.get("author"),
                "summary": f"[Error: {str(e)}]",
            }


async def summarize_chapters(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Summarize all chapters using Anthropic Batch API for 50% cost reduction.

    Uses batch processing for 5+ chapters, falls back to concurrent calls
    for smaller documents.

    Returns chapter_summaries list preserving chapter order.
    """
    try:
        markdown = state["processing_result"]["markdown"]
        chapters = state["chapters"]

        if not chapters:
            logger.warning("No chapters to summarize")
            return {
                "chapter_summaries": [],
                "current_status": "no_chapters",
            }

        # Use batch API for 5+ chapters (50% cost reduction)
        if len(chapters) >= 5:
            chapter_summaries = await _summarize_chapters_batched(chapters, markdown)
        else:
            # Fall back to concurrent calls for small documents
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHAPTER_SUMMARIES)
            tasks = [
                _summarize_single_chapter(
                    chapter=chapter,
                    chapter_content=markdown[chapter["start_position"]:chapter["end_position"]],
                    target_words=max(50, chapter["word_count"] // 10),
                    semaphore=semaphore,
                )
                for chapter in chapters
            ]
            logger.info(f"Starting concurrent summarization of {len(chapters)} chapters")
            chapter_summaries = await asyncio.gather(*tasks)

        logger.info(f"Completed summarization of {len(chapter_summaries)} chapters")

        return {
            "chapter_summaries": chapter_summaries,
            "current_status": "chapter_summaries_complete",
        }

    except Exception as e:
        logger.error(f"Failed to summarize chapters: {e}")
        return {
            "chapter_summaries": [],
            "current_status": "summarization_failed",
            "errors": [{"node": "summarize_chapters", "error": str(e)}],
        }


async def _summarize_chapters_batched(
    chapters: list[dict],
    markdown: str,
) -> list[dict[str, Any]]:
    """Summarize chapters using Anthropic Batch API for 50% cost reduction.

    Large chapters (>600k chars) are processed separately with chunking to avoid
    token limit errors, then results are merged in order.
    """
    processor = BatchProcessor(poll_interval=60)  # Longer poll for Opus

    # Separate normal and large chapters
    normal_indices = []
    large_indices = []
    chapter_data = []  # Store chapter info for result mapping

    for i, chapter in enumerate(chapters):
        chapter_content = markdown[chapter["start_position"]:chapter["end_position"]]
        target_words = max(50, chapter["word_count"] // 10)

        chapter_context = f"Chapter: {chapter['title']}"
        if chapter.get("author"):
            chapter_context += f" (by {chapter['author']})"

        chapter_data.append({
            "index": i,
            "title": chapter["title"],
            "author": chapter.get("author"),
            "content": chapter_content,
            "target_words": target_words,
            "context": chapter_context,
        })

        if len(chapter_content) > MAX_CHAPTER_CHARS:
            large_indices.append(i)
            logger.info(
                f"Chapter '{chapter['title']}' is large ({len(chapter_content)} chars), "
                "will process with chunking"
            )
        else:
            normal_indices.append(i)
            user_prompt = f"""Summarize this chapter in approximately {target_words} words.

Context: {chapter_context}

Chapter content:
{chapter_content}"""

            processor.add_request(
                custom_id=f"chapter-{i}",
                prompt=user_prompt,
                model=ModelTier.OPUS,
                max_tokens=8000 + 4096,
                system=CHAPTER_SUMMARIZATION_SYSTEM,
                thinking_budget=8000,
            )

    # Process normal chapters in batch
    batch_results = {}
    if normal_indices:
        logger.info(f"Submitting batch of {len(normal_indices)} normal-sized chapters")
        batch_results = await processor.execute_batch()

    # Process large chapters with chunking (concurrently)
    large_results = {}
    if large_indices:
        logger.info(f"Processing {len(large_indices)} large chapters with chunking")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHAPTER_SUMMARIES)
        tasks = []
        for idx in large_indices:
            data = chapter_data[idx]
            tasks.append(
                _summarize_single_chapter(
                    chapter={"title": data["title"], "author": data["author"]},
                    chapter_content=data["content"],
                    target_words=data["target_words"],
                    semaphore=semaphore,
                )
            )
        large_summaries = await asyncio.gather(*tasks)
        for idx, summary in zip(large_indices, large_summaries):
            large_results[idx] = summary

    # Merge results in original order
    chapter_summaries = []
    for i, chapter_info in enumerate(chapter_data):
        if i in large_results:
            # Large chapter was processed with chunking
            chapter_summaries.append(large_results[i])
        else:
            # Normal chapter was processed in batch
            result = batch_results.get(f"chapter-{i}")
            if result and result.success:
                summary = result.content
                if result.thinking:
                    logger.debug(f"Thinking for '{chapter_info['title']}': {result.thinking[:200]}...")
                chapter_summaries.append({
                    "title": chapter_info["title"],
                    "author": chapter_info["author"],
                    "summary": summary,
                })
            else:
                error_msg = result.error if result else "No result returned"
                logger.error(f"Failed to summarize chapter '{chapter_info['title']}': {error_msg}")
                chapter_summaries.append({
                    "title": chapter_info["title"],
                    "author": chapter_info["author"],
                    "summary": f"[Error: {error_msg}]",
                })

    return chapter_summaries


def aggregate_summaries(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Combine all chapter summaries into tenth_summary.

    Format: "## Chapter Title\n\nSummary text\n\n" for each chapter.

    Returns tenth_summary.
    """
    try:
        chapter_summaries = state.get("chapter_summaries", [])

        if not chapter_summaries:
            logger.warning("No chapter summaries to aggregate")
            return {
                "tenth_summary": "[No chapter summaries available]",
                "current_status": "aggregation_failed",
            }

        # Build combined summary
        parts = []
        for item in chapter_summaries:
            title = item["title"]
            author = item.get("author")
            summary = item["summary"]

            # Format header
            header = f"## {title}"
            if author:
                header += f" (by {author})"

            parts.append(f"{header}\n\n{summary}")

        tenth_summary = "\n\n".join(parts)

        logger.info(f"Aggregated {len(chapter_summaries)} chapter summaries into tenth summary")

        return {
            "tenth_summary": tenth_summary,
            "current_status": "tenth_summary_complete",
        }

    except Exception as e:
        logger.error(f"Failed to aggregate summaries: {e}")
        return {
            "tenth_summary": f"[Error aggregating summaries: {str(e)}]",
            "current_status": "aggregation_failed",
            "errors": [{"node": "aggregate_summaries", "error": str(e)}],
        }


def create_chapter_summarization_subgraph():
    """
    Create subgraph for parallel chapter summarization.

    Uses single node with asyncio.gather() for true concurrent execution:
    1. START -> summarize_chapters (batches all chapters concurrently)
    2. aggregate_summaries -> END
    """
    graph = StateGraph(DocumentProcessingState)

    # Add nodes
    graph.add_node("summarize_chapters", summarize_chapters)
    graph.add_node("aggregate_summaries", aggregate_summaries)

    # Add edges
    graph.add_edge(START, "summarize_chapters")
    graph.add_edge("summarize_chapters", "aggregate_summaries")
    graph.add_edge("aggregate_summaries", END)

    return graph.compile()


# Export compiled subgraph
chapter_summarization_subgraph = create_chapter_summarization_subgraph()
