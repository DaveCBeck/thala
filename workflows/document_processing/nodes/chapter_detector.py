"""
Chapter detection node for 10:1 summarization.
"""

import logging
import re
from typing import Any

from workflows.document_processing.state import ChapterInfo, DocumentProcessingState
from workflows.shared.llm_utils import extract_structured, ModelTier
from workflows.shared.text_utils import count_words

logger = logging.getLogger(__name__)

# Target chunk size for fallback chunking (in words)
FALLBACK_CHUNK_SIZE = 30000


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


def _create_fallback_chunks(markdown: str, word_count: int) -> list[ChapterInfo]:
    """
    Create pseudo-chapters by splitting document into ~30k word chunks.

    Used as fallback when heading-based chapter detection fails.
    Splits on paragraph boundaries to avoid breaking mid-sentence.
    """
    num_chunks = max(1, (word_count + FALLBACK_CHUNK_SIZE - 1) // FALLBACK_CHUNK_SIZE)
    target_chunk_size = len(markdown) // num_chunks

    chunks = []
    current_pos = 0

    for i in range(num_chunks):
        start_pos = current_pos

        if i == num_chunks - 1:
            # Last chunk gets everything remaining
            end_pos = len(markdown)
        else:
            # Find a good split point near target
            target_pos = start_pos + target_chunk_size
            # Look for paragraph break (double newline) near target
            search_start = max(start_pos, target_pos - 2000)
            search_end = min(len(markdown), target_pos + 2000)
            search_region = markdown[search_start:search_end]

            # Find last paragraph break in search region
            para_break = search_region.rfind("\n\n")
            if para_break != -1:
                end_pos = search_start + para_break + 2
            else:
                # No paragraph break, just split at target
                end_pos = target_pos

        chunk_text = markdown[start_pos:end_pos]
        chunk_word_count = count_words(chunk_text)

        chunks.append(ChapterInfo(
            title=f"Section {i + 1}",
            start_position=start_pos,
            end_position=end_pos,
            author=None,
            word_count=chunk_word_count,
        ))

        current_pos = end_pos

    logger.info(f"Created {len(chunks)} fallback chunks (~{FALLBACK_CHUNK_SIZE} words each)")
    return chunks


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

        # Only run 10:1 summary for documents with substantial content
        if word_count < 3000:
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

        # Schema for structured extraction (guaranteed valid JSON)
        schema = {
            "type": "object",
            "properties": {
                "headings": {
                    "type": "array",
                    "description": "Analysis of each heading",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string", "description": "Exact heading text"},
                            "is_chapter": {"type": "boolean", "description": "Whether this is a chapter boundary"},
                            "chapter_author": {"type": ["string", "null"], "description": "Author name if multi-author book"},
                        },
                        "required": ["heading", "is_chapter"],
                    },
                },
            },
            "required": ["headings"],
        }

        try:
            # Use structured extraction for guaranteed valid JSON
            result = await extract_structured(
                text=heading_list,
                prompt=prompt,
                schema=schema,
                tier=ModelTier.SONNET,
            )
            analysis = result.get("headings", [])

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
            # Graceful fallback: chunk document into ~30k word sections
            logger.warning(f"Chapter detection via LLM failed: {e}. Using fallback chunking.")
            chapters = _create_fallback_chunks(markdown, word_count)

            return {
                "chapters": chapters,
                "needs_tenth_summary": True,
                "current_status": "chapters_detected_fallback",
            }

    except Exception as e:
        # Outer exception handler for non-LLM failures (e.g., state issues)
        logger.error(f"Failed to detect chapters: {e}")
        return {
            "chapters": [],
            "needs_tenth_summary": False,
            "current_status": "chapter_detection_failed",
            "errors": [{"node": "detect_chapters", "error": str(e)}],
        }
