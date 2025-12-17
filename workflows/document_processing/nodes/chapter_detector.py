"""
Chapter detection node for 10:1 summarization.
"""

import logging
import re
from typing import Any

from workflows.document_processing.state import ChapterInfo, DocumentProcessingState
from workflows.shared.llm_utils import extract_json
from workflows.shared.text_utils import count_words

logger = logging.getLogger(__name__)


def _extract_headings(markdown: str) -> list[dict]:
    """
    Extract all headings from markdown with their positions.

    Returns list of {level, text, position} dicts.
    """
    headings = []
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    for match in heading_pattern.finditer(markdown):
        level = len(match.group(1))
        text = match.group(2).strip()
        position = match.start()

        headings.append({
            "level": level,
            "text": text,
            "position": position,
        })

    return headings


def _build_chapter_boundaries(
    markdown: str,
    headings: list[dict],
    analysis: list[dict]
) -> list[ChapterInfo]:
    """
    Build ChapterInfo list from heading analysis.

    Args:
        markdown: Full markdown text
        headings: List of all headings with positions
        analysis: LLM analysis marking chapter boundaries

    Returns:
        List of ChapterInfo dicts
    """
    # Find headings marked as chapters
    chapter_headings = []
    analysis_map = {item["heading"]: item for item in analysis}

    for heading in headings:
        heading_text = heading["text"]
        if heading_text in analysis_map and analysis_map[heading_text].get("is_chapter"):
            author = analysis_map[heading_text].get("chapter_author")
            chapter_headings.append({
                "title": heading_text,
                "position": heading["position"],
                "author": author,
            })

    # Build chapter boundaries
    chapters = []
    for i, chapter in enumerate(chapter_headings):
        start = chapter["position"]
        # End is start of next chapter, or end of document
        end = chapter_headings[i + 1]["position"] if i + 1 < len(chapter_headings) else len(markdown)

        chapter_text = markdown[start:end]
        word_count = count_words(chapter_text)

        chapters.append(ChapterInfo(
            title=chapter["title"],
            start_position=start,
            end_position=end,
            author=chapter["author"],
            word_count=word_count,
        ))

    return chapters


async def detect_chapters(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Use LLM to analyze heading structure and detect logical chapter divisions.

    Process:
    1. Extract all headings from markdown
    2. Send heading list to LLM to identify chapter-level divisions
    3. For multi-author books, identify chapter authors from metadata_updates
    4. Build ChapterInfo list with positions and word counts

    If no headings found, treat entire document as single chapter.
    If document is short (<50k words), skip chapter detection and 10:1 summary.

    Returns chapters list, needs_tenth_summary flag, and current_status.
    """
    try:
        processing_result = state.get("processing_result")
        if not processing_result:
            logger.error("No processing_result in state")
            return {
                "chapters": [],
                "needs_tenth_summary": False,
                "current_status": "chapter_detection_failed",
                "errors": [{"node": "detect_chapters", "error": "No processing result"}],
            }

        markdown = processing_result["markdown"]
        word_count = processing_result.get("word_count", count_words(markdown))

        # Only run 10:1 summary for long documents
        if word_count < 50000:
            logger.info(f"Document too short ({word_count} words), skipping 10:1 summary")
            return {
                "chapters": [],
                "needs_tenth_summary": False,
                "current_status": "chapter_detection_skipped",
            }

        # Extract headings
        headings = _extract_headings(markdown)

        # If no headings, treat entire document as single chapter
        if not headings:
            logger.info("No headings found, treating as single chapter")
            chapters = [ChapterInfo(
                title="Full Document",
                start_position=0,
                end_position=len(markdown),
                author=None,
                word_count=word_count,
            )]
            return {
                "chapters": chapters,
                "needs_tenth_summary": True,
                "current_status": "chapters_detected",
            }

        # Check if metadata indicates multi-author book
        metadata_updates = state.get("metadata_updates", {})
        is_multi_author = metadata_updates.get("multi_author", False)

        # Prepare heading list for LLM
        heading_list = "\n".join([
            f"{'#' * h['level']} {h['text']}"
            for h in headings
        ])

        # Build prompt
        prompt = """Analyze this list of document headings and identify which ones represent chapter-level divisions.
Mark each heading with is_chapter=true if it represents a chapter boundary, false otherwise.

Guidelines:
- Look for consistent patterns (e.g., all H1s, or "Chapter N" patterns)
- Chapters should be major divisions of the document
- Sub-sections within chapters should be marked false"""

        if is_multi_author:
            prompt += "\n\nThis is a multi-author book. For each chapter, identify the author name if present in the heading."

        # Build schema hint
        schema_hint = """[
  {
    "heading": "exact heading text",
    "is_chapter": true,
    "chapter_author": "Author Name or null"
  }
]"""

        # Call LLM
        analysis = await extract_json(
            text=heading_list,
            prompt=prompt,
            schema_hint=schema_hint,
        )

        # Build chapter boundaries
        chapters = _build_chapter_boundaries(markdown, headings, analysis)

        # If no chapters identified, treat entire document as single chapter
        if not chapters:
            logger.warning("No chapters identified by LLM, treating as single chapter")
            chapters = [ChapterInfo(
                title="Full Document",
                start_position=0,
                end_position=len(markdown),
                author=None,
                word_count=word_count,
            )]

        logger.info(f"Detected {len(chapters)} chapters for 10:1 summary")
        return {
            "chapters": chapters,
            "needs_tenth_summary": True,
            "current_status": "chapters_detected",
        }

    except Exception as e:
        logger.error(f"Failed to detect chapters: {e}")
        return {
            "chapters": [],
            "needs_tenth_summary": False,
            "current_status": "chapter_detection_failed",
            "errors": [{"node": "detect_chapters", "error": str(e)}],
        }
