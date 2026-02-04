"""Node implementations for chapter summarization.

Routes through central LLM broker for unified cost/speed management.
"""

import asyncio
import logging
from typing import Any

from langsmith import traceable

from core.config import truncate_for_trace
from core.llm_broker import BatchPolicy, get_broker
from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier
from workflows.shared.retry_utils import with_retry
from workflows.shared.token_utils import HAIKU_SAFE_LIMIT, estimate_tokens_fast

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
    """Summarize a single chunk of content via broker."""
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
    estimated_tokens = estimate_tokens_fast(user_prompt + CHAPTER_SUMMARIZATION_SYSTEM)

    if estimated_tokens > HAIKU_SAFE_LIMIT:
        logger.info(
            f"Content exceeds Haiku safe limit ({estimated_tokens:,} > {HAIKU_SAFE_LIMIT:,} tokens), using SONNET_1M"
        )
        model_tier = ModelTier.SONNET_1M
    else:
        model_tier = ModelTier.HAIKU

    # TODO: Upgrade to ModelTier.OPUS before production
    broker = get_broker()
    future = await broker.request(
        prompt=user_prompt,
        model=model_tier,
        policy=BatchPolicy.PREFER_BALANCE,
        max_tokens=8000 + 4096,
        system=CHAPTER_SUMMARIZATION_SYSTEM,
        thinking_budget=8000,
    )
    response = await future

    if not response.success:
        logger.warning(f"Chunk summarization failed: {response.error}")
        return f"[Summarization failed: {response.error}]"

    return response.content


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


@traceable(
    run_type="chain",
    name="SummarizeChapters",
    process_inputs=truncate_for_trace,
    process_outputs=truncate_for_trace,
)
async def summarize_chapters(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Summarize all chapters via central LLM broker.

    Routes through broker for unified cost/speed management.
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

        broker = get_broker()
        pending_futures: dict[int, tuple[dict, asyncio.Future | None, str | None]] = {}

        # Separate normal and large chapters
        # Large chapters need chunking and can't go through simple broker path
        large_chapter_indices = set()

        logger.info(f"Submitting {len(chapters)} chapters for summarization via broker")

        async with broker.batch_group():
            for i, chapter in enumerate(chapters):
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
                    pending_futures[i] = (chapter, None, None)
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
                estimated_tokens = estimate_tokens_fast(user_prompt + CHAPTER_SUMMARIZATION_SYSTEM)

                if estimated_tokens > HAIKU_SAFE_LIMIT:
                    # Too many tokens for Haiku - process individually with SONNET_1M
                    large_chapter_indices.add(i)
                    logger.info(
                        f"Chapter '{chapter['title']}' exceeds Haiku token limit "
                        f"({estimated_tokens:,} > {HAIKU_SAFE_LIMIT:,} tokens), "
                        "will process individually with SONNET_1M"
                    )
                    pending_futures[i] = (chapter, None, None)
                    continue

                # Normal chapter - submit to broker
                # TODO: Upgrade to ModelTier.OPUS before production
                future = await broker.request(
                    prompt=user_prompt,
                    model=ModelTier.HAIKU,
                    policy=BatchPolicy.PREFER_BALANCE,
                    max_tokens=8000 + 4096,
                    system=CHAPTER_SUMMARIZATION_SYSTEM,
                    thinking_budget=8000,
                )
                pending_futures[i] = (chapter, future, chapter_content)

        # Process large chapters with chunking (these can't be batched)
        large_results = {}
        for idx in large_chapter_indices:
            chapter, _, _ = pending_futures[idx]
            chapter_content = markdown[chapter["start_position"] : chapter["end_position"]]
            target_words = max(50, chapter["word_count"] // 10)
            result = await _summarize_single_chapter(
                chapter=chapter,
                chapter_content=chapter_content,
                target_words=target_words,
            )
            large_results[idx] = result

        # Collect results in order
        chapter_summaries = []
        for i in range(len(chapters)):
            if i in large_results:
                chapter_summaries.append(large_results[i])
            else:
                chapter, future, chapter_content = pending_futures[i]
                try:
                    response = await future
                    if response.success:
                        summary = response.content
                        chapter_summaries.append(
                            {
                                "title": chapter["title"],
                                "author": chapter.get("author"),
                                "summary": summary,
                            }
                        )
                    else:
                        logger.error(f"Failed to summarize chapter '{chapter['title']}': {response.error}")
                        chapter_summaries.append(
                            {
                                "title": chapter["title"],
                                "author": chapter.get("author"),
                                "summary": f"[Error: {response.error}]",
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to summarize chapter '{chapter['title']}': {e}")
                    chapter_summaries.append(
                        {
                            "title": chapter["title"],
                            "author": chapter.get("author"),
                            "summary": f"[Error: {e}]",
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
    """Translate text to English using Sonnet via broker."""

    async def _invoke():
        broker = get_broker()
        future = await broker.request(
            prompt=f"Translate this text to English:\n\n{text}",
            model=ModelTier.SONNET,
            policy=BatchPolicy.PREFER_BALANCE,
            max_tokens=len(text) // 2 + 1000,  # Translation similar length
            system=TRANSLATION_SYSTEM,
        )
        response = await future
        if response.success:
            return response.content
        raise RuntimeError(f"Translation failed: {response.error}")

    return await with_retry(_invoke)
