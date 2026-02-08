"""Node implementations for chapter summarization.

Routes through unified invoke() for automatic broker routing and cost optimization.
"""

import asyncio
import logging
from typing import Any

from langsmith import traceable

from core.config import truncate_for_trace
from core.llm_broker import BatchPolicy
from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, invoke, invoke_batch, InvokeConfig
from workflows.shared.retry_utils import with_retry
from workflows.shared.token_utils import HAIKU_SAFE_LIMIT, count_tokens_accurate

from .chunking import MAX_CHAPTER_CHARS, chunk_large_content
from .prompts import CHAPTER_SUMMARIZATION_SYSTEM, TRANSLATION_SYSTEM

logger = logging.getLogger(__name__)


async def _summarize_content_chunk(
    content: str,
    target_words: int,
    chapter_context: str,
    chunk_num: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """Summarize a single chunk of content via invoke()."""
    chunk_info = ""
    if chunk_num is not None and total_chunks is not None:
        chunk_info = f" (Part {chunk_num}/{total_chunks})"

    # Short summaries should be prose-only without headings
    if target_words < 800:
        format_instruction = " Use text-only prose with no headings."
    else:
        format_instruction = ""

    user_prompt = f"""Summarize this content in approximately {target_words} words.{format_instruction}

Context: {chapter_context}{chunk_info}

Content:
{content}"""

    # Estimate tokens to select appropriate model
    # Use SONNET_1M for large content to avoid token limit errors
    estimated_tokens = count_tokens_accurate(user_prompt + CHAPTER_SUMMARIZATION_SYSTEM)

    if estimated_tokens > HAIKU_SAFE_LIMIT:
        logger.info(
            f"Content exceeds Haiku safe limit ({estimated_tokens:,} > {HAIKU_SAFE_LIMIT:,} tokens), using SONNET_1M"
        )
        model_tier = ModelTier.SONNET_1M
    else:
        model_tier = ModelTier.HAIKU

    # TODO: Upgrade to ModelTier.OPUS before production
    try:
        response = await invoke(
            tier=model_tier,
            system=CHAPTER_SUMMARIZATION_SYSTEM,
            user=user_prompt,
            config=InvokeConfig(
                batch_policy=BatchPolicy.PREFER_BALANCE,
                max_tokens=8000 + 4096,
                thinking_budget=8000,
                cache=False,  # Required when using thinking_budget
            ),
        )
        return response.content
    except Exception as e:
        logger.warning(f"Chunk summarization failed: {e}")
        return f"[Summarization failed: {e}]"


async def _summarize_single_chapter(
    chapter: dict,
    chapter_content: str,
    target_words: int,
) -> dict[str, Any]:
    """
    Summarize a single chapter to ~10% of original length.

    Uses extended thinking for deep analysis of chapter content,
    including title and author context.

    For very long chapters (>600k chars), chunks the content and summarizes each
    chunk separately, then combines the summaries.

    Returns dict with title, author, and summary.
    """
    try:
        # Build context for the chapter
        chapter_context = f"Chapter: {chapter['title']}"
        if chapter.get("author"):
            chapter_context += f" (by {chapter['author']})"

        # Check if content needs chunking
        chunks = chunk_large_content(chapter_content)

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

            chunk_summaries = await asyncio.gather(*(
                _summarize_content_chunk(
                    content=chunk,
                    target_words=chunk_target_words,
                    chapter_context=chapter_context,
                    chunk_num=i,
                    total_chunks=len(chunks),
                )
                for i, chunk in enumerate(chunks, 1)
            ))

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


@traceable(
    run_type="chain",
    name="SummarizeChapters",
    process_inputs=truncate_for_trace,
    process_outputs=truncate_for_trace,
)
async def summarize_chapters(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Summarize all chapters via invoke_batch().

    Uses invoke_batch() for efficient batched LLM calls with automatic
    broker routing and cost optimization.
    Large chapters (>600k chars) are processed with chunking.

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

        # Track chapter metadata for result collection
        batch_chapter_info: list[tuple[int, dict]] = []  # (index, chapter)
        large_chapter_indices: set[int] = set()
        chapters_metadata: dict[int, dict] = {}  # index -> chapter

        logger.info(f"Submitting {len(chapters)} chapters for summarization")

        # Use invoke_batch for efficient batching
        async with invoke_batch() as batch:
            for i, chapter in enumerate(chapters):
                chapters_metadata[i] = chapter
                chapter_content = markdown[chapter["start_position"] : chapter["end_position"]]
                target_words = max(50, chapter["word_count"] // 10)

                chapter_context = f"Chapter: {chapter['title']}"
                if chapter.get("author"):
                    chapter_context += f" (by {chapter['author']})"

                # Check if needs chunking (by character count) or SONNET_1M (by token count)
                if len(chapter_content) > MAX_CHAPTER_CHARS:
                    large_chapter_indices.add(i)
                    logger.info(
                        f"Chapter '{chapter['title']}' is large ({len(chapter_content)} chars), "
                        "will process with chunking"
                    )
                    continue

                # Short summaries should be prose-only without headings
                if target_words < 800:
                    format_instruction = " Use text-only prose with no headings."
                else:
                    format_instruction = ""

                user_prompt = f"""Summarize this chapter in approximately {target_words} words.{format_instruction}

Context: {chapter_context}

Chapter content:
{chapter_content}"""

                # Check token count
                estimated_tokens = count_tokens_accurate(user_prompt + CHAPTER_SUMMARIZATION_SYSTEM)

                if estimated_tokens > HAIKU_SAFE_LIMIT:
                    # Too many tokens for Haiku - process individually with SONNET_1M
                    large_chapter_indices.add(i)
                    logger.info(
                        f"Chapter '{chapter['title']}' exceeds Haiku token limit "
                        f"({estimated_tokens:,} > {HAIKU_SAFE_LIMIT:,} tokens), "
                        "will process individually with SONNET_1M"
                    )
                    continue

                # Normal chapter - add to batch
                # TODO: Upgrade to ModelTier.OPUS before production
                batch.add(
                    tier=ModelTier.HAIKU,
                    system=CHAPTER_SUMMARIZATION_SYSTEM,
                    user=user_prompt,
                    config=InvokeConfig(
                        batch_policy=BatchPolicy.PREFER_BALANCE,
                        max_tokens=8000 + 4096,
                        thinking_budget=8000,
                        cache=False,  # Required when using thinking_budget
                    ),
                )
                batch_chapter_info.append((i, chapter))

        # Collect batch results
        batch_results = await batch.results()

        # Build index -> result mapping from batch
        batch_result_map: dict[int, str] = {}
        for batch_idx, (chapter_idx, chapter) in enumerate(batch_chapter_info):
            try:
                response = batch_results[batch_idx]
                batch_result_map[chapter_idx] = response.content
            except Exception as e:
                logger.error(f"Failed to summarize chapter '{chapter['title']}': {e}")
                batch_result_map[chapter_idx] = f"[Error: {e}]"

        # Process large chapters with chunking concurrently (these can't be batched)
        large_results: dict[int, dict] = {}
        if large_chapter_indices:
            large_items = sorted(large_chapter_indices)
            large_coros = []
            for idx in large_items:
                chapter = chapters_metadata[idx]
                chapter_content = markdown[chapter["start_position"] : chapter["end_position"]]
                target_words = max(50, chapter["word_count"] // 10)
                large_coros.append(_summarize_single_chapter(
                    chapter=chapter,
                    chapter_content=chapter_content,
                    target_words=target_words,
                ))
            results = await asyncio.gather(*large_coros)
            for idx, result in zip(large_items, results):
                large_results[idx] = result

        # Collect results in order
        chapter_summaries = []
        for i in range(len(chapters)):
            chapter = chapters_metadata[i]
            if i in large_results:
                chapter_summaries.append(large_results[i])
            elif i in batch_result_map:
                chapter_summaries.append(
                    {
                        "title": chapter["title"],
                        "author": chapter.get("author"),
                        "summary": batch_result_map[i],
                    }
                )
            else:
                # Should not happen, but handle gracefully
                logger.error(f"No result for chapter '{chapter['title']}'")
                chapter_summaries.append(
                    {
                        "title": chapter["title"],
                        "author": chapter.get("author"),
                        "summary": "[Error: No result available]",
                    }
                )

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


@traceable(
    run_type="chain",
    name="AggregateSummaries",
    process_inputs=truncate_for_trace,
    process_outputs=truncate_for_trace,
)
async def aggregate_summaries(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Combine all chapter summaries into tenth_summary.

    Format: "## Chapter Title\n\nSummary text\n\n" for each chapter.
    For non-English docs: also generates English translation.

    Returns tenth_summary, tenth_summary_original, and tenth_summary_english.
    """
    try:
        chapter_summaries = state.get("chapter_summaries", [])
        original_language = state.get("original_language", "en")

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

        result = {
            "tenth_summary": tenth_summary,  # Backward compatibility
            "tenth_summary_original": tenth_summary,
            "current_status": "tenth_summary_complete",
        }

        # If non-English, also generate English translation
        if original_language != "en":
            english_summary = await _translate_to_english(tenth_summary)
            result["tenth_summary_english"] = english_summary
            logger.info(f"Generated English translation of tenth summary ({len(english_summary.split())} words)")

        return result

    except Exception as e:
        logger.error(f"Failed to aggregate summaries: {e}")
        return {
            "tenth_summary": f"[Error aggregating summaries: {str(e)}]",
            "current_status": "aggregation_failed",
            "errors": [{"node": "aggregate_summaries", "error": str(e)}],
        }


async def _translate_to_english(text: str) -> str:
    """Translate text to English using Sonnet via invoke()."""

    async def _invoke_translation():
        response = await invoke(
            tier=ModelTier.SONNET,
            system=TRANSLATION_SYSTEM,
            user=f"Translate this text to English:\n\n{text}",
            config=InvokeConfig(
                batch_policy=BatchPolicy.PREFER_BALANCE,
                max_tokens=len(text) // 2 + 1000,  # Translation similar length
            ),
        )
        return response.content

    return await with_retry(_invoke_translation)
