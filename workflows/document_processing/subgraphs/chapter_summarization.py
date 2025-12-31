"""
Chapter summarization subgraph using map-reduce pattern.

Uses Opus with extended thinking for complex chapter analysis.
Implements prompt caching for 90% cost reduction on static instructions.
"""

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from workflows.document_processing.state import ChapterSummaryState, DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache

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


async def summarize_chapter(state: ChapterSummaryState) -> dict[str, Any]:
    """
    Summarize a single chapter to ~10% of original length.

    Uses Opus with extended thinking for deep analysis of chapter content,
    including title and author context. Implements prompt caching for cost reduction.

    Returns summary and appends to chapter_summaries list.
    """
    try:
        chapter = state["chapter"]
        chapter_content = state["chapter_content"]
        target_words = state["target_words"]

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

        # Return summary for aggregation (only chapter_summaries, not summary)
        # Note: We don't return "summary" because DocumentProcessingState doesn't have
        # that field, and parallel Send() calls would cause InvalidUpdateError
        return {
            "chapter_summaries": [{
                "title": chapter["title"],
                "author": chapter.get("author"),
                "summary": summary,
            }],
        }

    except Exception as e:
        logger.error(f"Failed to summarize chapter '{chapter['title']}': {e}")
        # Return error in chapter_summaries only (not summary field)
        return {
            "chapter_summaries": [{
                "title": chapter["title"],
                "author": chapter.get("author"),
                "summary": f"[Error: {str(e)}]",
            }],
        }


def fan_out_chapters(state: DocumentProcessingState):
    """
    Fan out to parallel chapter summarization tasks.

    Creates a Send() for each chapter with its content and target word count.
    """
    markdown = state["processing_result"]["markdown"]
    chapters = state["chapters"]

    return [
        Send("summarize_chapter", {
            "chapter": chapter,
            "chapter_content": markdown[chapter["start_position"]:chapter["end_position"]],
            "target_words": max(50, chapter["word_count"] // 10),
        })
        for chapter in chapters
    ]


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

    Uses map-reduce pattern:
    1. START -> fan_out_chapters (creates Send per chapter)
    2. summarize_chapter (parallel per chapter)
    3. aggregate_summaries -> END
    """
    graph = StateGraph(DocumentProcessingState)

    # Add nodes
    graph.add_node("summarize_chapter", summarize_chapter)
    graph.add_node("aggregate_summaries", aggregate_summaries)

    # Add edges
    graph.add_conditional_edges(START, fan_out_chapters, ["summarize_chapter"])
    graph.add_edge("summarize_chapter", "aggregate_summaries")
    graph.add_edge("aggregate_summaries", END)

    return graph.compile()


# Export compiled subgraph
chapter_summarization_subgraph = create_chapter_summarization_subgraph()
