"""
Chapter summarization subgraph using map-reduce pattern.

Uses Opus with extended thinking for complex chapter analysis.
Uses Anthropic Batch API for 50% cost reduction when processing 5+ chapters.
"""

import asyncio
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

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

    Returns dict with title, author, and summary.
    """
    async with semaphore:
        try:
            # Build context for the chapter
            chapter_context = f"Chapter: {chapter['title']}"
            if chapter.get("author"):
                chapter_context += f" (by {chapter['author']})"

            # Build dynamic user prompt
            user_prompt = f"""Summarize this chapter in approximately {target_words} words.

Context: {chapter_context}

Chapter content:
{chapter_content}"""

            # Use Opus with extended thinking for complex chapter analysis
            # Note: Extended thinking is enabled on the LLM, system prompt is cached
            llm = get_llm(
                tier=ModelTier.OPUS,
                thinking_budget=8000,
                max_tokens=8000 + 4096,
            )

            # Use cached system prompt for 90% cost reduction
            response = await invoke_with_cache(
                llm,
                system_prompt=CHAPTER_SUMMARIZATION_SYSTEM,  # ~200 tokens, cached
                user_prompt=user_prompt,  # Dynamic content
            )

            # Extract text content from response (may include thinking blocks)
            thinking = None
            if isinstance(response.content, list):
                summary = ""
                for block in response.content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            thinking = block.get("thinking", "")
                        elif block.get("type") == "text":
                            summary = block.get("text", "")
                    elif hasattr(block, "type"):
                        if block.type == "thinking":
                            thinking = getattr(block, "thinking", "")
                        elif block.type == "text":
                            summary = getattr(block, "text", "")
            else:
                summary = response.content

            logger.info(f"Summarized chapter '{chapter['title']}' to {len(summary.split())} words")
            if thinking:
                logger.debug(f"Thinking summary for '{chapter['title']}': {thinking[:200]}...")

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
    """Summarize chapters using Anthropic Batch API for 50% cost reduction."""
    processor = BatchProcessor(poll_interval=60)  # Longer poll for Opus

    chapter_data = []  # Store chapter info for result mapping
    for i, chapter in enumerate(chapters):
        chapter_content = markdown[chapter["start_position"]:chapter["end_position"]]
        target_words = max(50, chapter["word_count"] // 10)

        chapter_context = f"Chapter: {chapter['title']}"
        if chapter.get("author"):
            chapter_context += f" (by {chapter['author']})"

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

        chapter_data.append({
            "title": chapter["title"],
            "author": chapter.get("author"),
        })

    logger.info(f"Submitting batch of {len(chapters)} chapters for summarization")
    results = await processor.execute_batch()

    chapter_summaries = []
    for i, chapter_info in enumerate(chapter_data):
        result = results.get(f"chapter-{i}")
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
