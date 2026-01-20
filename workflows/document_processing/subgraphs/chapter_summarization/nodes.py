"""Node implementations for chapter summarization."""

import asyncio
import logging
from typing import Any

from langsmith import traceable

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from workflows.shared.llm_utils.response_parsing import extract_response_content
from workflows.shared.batch_processor import BatchProcessor
from workflows.shared.retry_utils import with_retry
from workflows.shared.token_utils import estimate_tokens_fast, HAIKU_SAFE_LIMIT

from .chunking import chunk_large_content, MAX_CHAPTER_CHARS
from .prompts import CHAPTER_SUMMARIZATION_SYSTEM, TRANSLATION_SYSTEM

logger = logging.getLogger(__name__)

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

    # Estimate tokens to select appropriate model
    # Use SONNET_1M for large content to avoid token limit errors
    estimated_tokens = estimate_tokens_fast(user_prompt + CHAPTER_SUMMARIZATION_SYSTEM)

    if estimated_tokens > HAIKU_SAFE_LIMIT:
        logger.info(
            f"Content exceeds Haiku safe limit ({estimated_tokens:,} > {HAIKU_SAFE_LIMIT:,} tokens), "
            f"using SONNET_1M"
        )
        model_tier = ModelTier.SONNET_1M
    else:
        model_tier = ModelTier.HAIKU

    # TODO: Upgrade to ModelTier.OPUS before production
    llm = get_llm(
        tier=model_tier,
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

            logger.info(
                f"Summarized chapter '{chapter['title']}' to {len(summary.split())} words"
            )

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


@traceable(run_type="chain", name="SummarizeChapters")
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
        use_batch_api = state["input"].get("use_batch_api", True)

        if not chapters:
            logger.warning("No chapters to summarize")
            return {
                "chapter_summaries": [],
                "current_status": "no_chapters",
            }

        # Use batch API for 5+ chapters (50% cost reduction) when enabled
        if use_batch_api and len(chapters) >= 5:
            chapter_summaries = await _summarize_chapters_batched(chapters, markdown)
        else:
            # Fall back to concurrent calls for small documents
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHAPTER_SUMMARIES)
            tasks = [
                _summarize_single_chapter(
                    chapter=chapter,
                    chapter_content=markdown[
                        chapter["start_position"] : chapter["end_position"]
                    ],
                    target_words=max(50, chapter["word_count"] // 10),
                    semaphore=semaphore,
                )
                for chapter in chapters
            ]
            logger.info(
                f"Starting concurrent summarization of {len(chapters)} chapters"
            )
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
        chapter_content = markdown[chapter["start_position"] : chapter["end_position"]]
        target_words = max(50, chapter["word_count"] // 10)

        chapter_context = f"Chapter: {chapter['title']}"
        if chapter.get("author"):
            chapter_context += f" (by {chapter['author']})"

        chapter_data.append(
            {
                "index": i,
                "title": chapter["title"],
                "author": chapter.get("author"),
                "content": chapter_content,
                "target_words": target_words,
                "context": chapter_context,
            }
        )

        # Build the prompt to check token count
        user_prompt = f"""Summarize this chapter in approximately {target_words} words.

Context: {chapter_context}

Chapter content:
{chapter_content}"""

        # Check if needs chunking (by character count) or SONNET_1M (by token count)
        estimated_tokens = estimate_tokens_fast(user_prompt + CHAPTER_SUMMARIZATION_SYSTEM)

        if len(chapter_content) > MAX_CHAPTER_CHARS:
            large_indices.append(i)
            logger.info(
                f"Chapter '{chapter['title']}' is large ({len(chapter_content)} chars), "
                "will process with chunking"
            )
        elif estimated_tokens > HAIKU_SAFE_LIMIT:
            # Too many tokens for Haiku batch API - process individually with SONNET_1M
            large_indices.append(i)
            logger.info(
                f"Chapter '{chapter['title']}' exceeds Haiku token limit "
                f"({estimated_tokens:,} > {HAIKU_SAFE_LIMIT:,} tokens), "
                "will process individually with SONNET_1M"
            )
        else:
            normal_indices.append(i)

            # TODO: Upgrade to ModelTier.OPUS before production
            processor.add_request(
                custom_id=f"chapter-{i}",
                prompt=user_prompt,
                model=ModelTier.HAIKU,
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
                    logger.debug(
                        f"Thinking for '{chapter_info['title']}': {result.thinking[:200]}..."
                    )
                chapter_summaries.append(
                    {
                        "title": chapter_info["title"],
                        "author": chapter_info["author"],
                        "summary": summary,
                    }
                )
            else:
                error_msg = result.error if result else "No result returned"
                logger.error(
                    f"Failed to summarize chapter '{chapter_info['title']}': {error_msg}"
                )
                chapter_summaries.append(
                    {
                        "title": chapter_info["title"],
                        "author": chapter_info["author"],
                        "summary": f"[Error: {error_msg}]",
                    }
                )

    return chapter_summaries


@traceable(run_type="chain", name="AggregateSummaries")
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

        logger.info(
            f"Aggregated {len(chapter_summaries)} chapter summaries into tenth summary"
        )

        result = {
            "tenth_summary": tenth_summary,  # Backward compatibility
            "tenth_summary_original": tenth_summary,
            "current_status": "tenth_summary_complete",
        }

        # If non-English, also generate English translation
        if original_language != "en":
            english_summary = await _translate_to_english(tenth_summary)
            result["tenth_summary_english"] = english_summary
            logger.info(
                f"Generated English translation of tenth summary ({len(english_summary.split())} words)"
            )

        return result

    except Exception as e:
        logger.error(f"Failed to aggregate summaries: {e}")
        return {
            "tenth_summary": f"[Error aggregating summaries: {str(e)}]",
            "current_status": "aggregation_failed",
            "errors": [{"node": "aggregate_summaries", "error": str(e)}],
        }


async def _translate_to_english(text: str) -> str:
    """Translate text to English using Sonnet."""
    llm = get_llm(tier=ModelTier.SONNET)

    async def _invoke():
        response = await invoke_with_cache(
            llm,
            system_prompt=TRANSLATION_SYSTEM,
            user_prompt=f"Translate this text to English:\n\n{text}",
        )
        return extract_response_content(response)

    return await with_retry(_invoke)
