"""
Chapter detection node for 10:1 summarization.
"""

import json
import logging
from typing import Any, Optional

from langsmith import traceable
from pydantic import BaseModel, Field, field_validator

from workflows.document_processing.state import ChapterInfo, DocumentProcessingState
from workflows.shared.chunking_utils import (
    create_fallback_chunks,
    create_heading_based_chapters,
)
from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier
from workflows.shared.markdown_utils import extract_headings
from workflows.shared.text_utils import count_words
from workflows.shared.token_utils import estimate_tokens_fast

logger = logging.getLogger(__name__)


class HeadingAnalysis(BaseModel):
    """Analysis of a single heading."""

    heading: str = Field(description="Exact heading text (without # prefix or token count)")
    is_content: bool = Field(
        description="True for substantive content, false for non-content (Abstract, References, Acknowledgements, etc.)"
    )
    chunk_boundary: bool = Field(description="True if this heading starts a new chunk for summarization")
    chapter_author: Optional[str] = Field(default=None, description="Author name if multi-author book")


class HeadingAnalysisResult(BaseModel):
    """Result of heading structure analysis."""

    headings: list[HeadingAnalysis] = Field(description="Analysis of each heading")

    @field_validator("headings", mode="before")
    @classmethod
    def parse_json_string(cls, v: Any) -> list:
        """Handle LLM returning JSON string instead of list.

        This addresses a known issue where Claude's structured output sometimes
        returns arrays as stringified JSON rather than proper array structures.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            raise ValueError(f"headings must be a list, got unparseable string: {v[:100]}...")
        return v if v is not None else []


NON_CONTENT_HEADINGS = {
    "abstract", "references", "bibliography", "works cited",
    "acknowledgements", "acknowledgments", "author contributions",
    "funding", "conflicts of interest", "conflict of interest",
    "abbreviations", "appendix", "supplementary material",
    "supplementary data", "supporting information",
}


def _is_non_content_heading(text: str) -> bool:
    """Check if a heading is non-content (references, acknowledgements, etc.)."""
    normalised = text.strip("*").strip().lower()
    return normalised in NON_CONTENT_HEADINGS


def _build_chapter_boundaries(markdown: str, headings: list[dict], analysis: list[dict]) -> list[ChapterInfo]:
    """
    Build ChapterInfo list from heading analysis.

    Args:
        markdown: Full markdown text
        headings: List of all headings with positions
        analysis: LLM analysis with is_content and chunk_boundary fields

    Returns:
        List of ChapterInfo dicts
    """
    analysis_map = {item["heading"]: item for item in analysis}

    # Collect headings where both is_content=true AND chunk_boundary=true
    chapter_headings = []
    for heading in headings:
        heading_text = heading["text"]
        item = analysis_map.get(heading_text)
        if item and item.get("is_content") and item.get("chunk_boundary"):
            chapter_headings.append(
                {
                    "title": heading_text,
                    "position": heading["position"],
                    "author": item.get("chapter_author"),
                }
            )

    # Find the position of the first non-content heading after the last chapter.
    # Uses LLM-driven is_content=false classification, with _is_non_content_heading as fallback.
    first_non_content_pos = None
    if chapter_headings:
        last_chapter_pos = chapter_headings[-1]["position"]
        for heading in headings:
            if heading["position"] <= last_chapter_pos:
                continue
            heading_text = heading["text"]
            item = analysis_map.get(heading_text)
            is_non_content = (item and not item.get("is_content")) or _is_non_content_heading(heading_text)
            if is_non_content:
                first_non_content_pos = heading["position"]
                break

    # Build chapter boundaries
    chapters = []
    for i, chapter in enumerate(chapter_headings):
        start = chapter["position"]
        if i + 1 < len(chapter_headings):
            end = chapter_headings[i + 1]["position"]
        elif first_non_content_pos is not None:
            end = first_non_content_pos
        else:
            end = len(markdown)

        chapter_text = markdown[start:end]
        word_count = count_words(chapter_text)

        chapters.append(
            ChapterInfo(
                title=chapter["title"],
                start_position=start,
                end_position=end,
                author=chapter["author"],
                word_count=word_count,
            )
        )

    return chapters


@traceable(run_type="chain", name="DetectChapters")
async def detect_chapters(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Use LLM to analyze heading structure and detect logical chapter divisions.

    Process:
    1. Extract all headings from markdown
    2. Send heading list to LLM to identify chapter-level divisions
    3. For multi-author books, identify chapter authors from metadata_updates
    4. Build ChapterInfo list with positions and word counts

    If no headings found, use fallback chunking into ~30k word sections.
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

        # Log document token count for tracking
        doc_tokens = estimate_tokens_fast(markdown, with_safety_margin=False)
        logger.info(f"[TOKEN_TRACKING] Document content: {doc_tokens:,} tokens ({word_count:,} words)")

        # Only run 10:1 summary for documents with substantial content
        if word_count < 3000:
            logger.info(f"Document too short ({word_count} words), skipping 10:1 summary")
            return {
                "chapters": [],
                "needs_tenth_summary": False,
                "current_status": "chapter_detection_skipped",
            }

        # Extract headings
        headings = extract_headings(markdown)

        # Log heading structure for debugging
        if headings:
            heading_counts = {}
            for h in headings:
                level = h["level"]
                heading_counts[level] = heading_counts.get(level, 0) + 1
            logger.debug(
                f"Found {len(headings)} headings: "
                f"{', '.join(f'H{lvl}={cnt}' for lvl, cnt in sorted(heading_counts.items()))}"
            )

        # If no headings, use fallback chunking
        if not headings:
            logger.info("No headings found, using fallback chunking")
            chapters = create_fallback_chunks(markdown, word_count, ChapterInfo)
            return {
                "chapters": chapters,
                "needs_tenth_summary": True,
                "current_status": "chapters_detected_fallback",
            }

        # Check if metadata indicates multi-author book
        metadata_updates = state.get("metadata_updates", {})
        is_multi_author = metadata_updates.get("multi_author", False)

        # Prepare heading list for LLM — include token counts per section
        heading_lines = []
        for i, h in enumerate(headings):
            start = h["position"]
            end = headings[i + 1]["position"] if i + 1 < len(headings) else len(markdown)
            section_tokens = estimate_tokens_fast(markdown[start:end], with_safety_margin=False)
            heading_lines.append(f"{'#' * h['level']} {h['text']}  (~{section_tokens:,} tokens)")
        heading_list = "\n".join(heading_lines)

        # Build prompt
        system_prompt = """You are splitting a document into chapters for two downstream purposes:
1. Each chapter will be summarized independently (10:1 compression).
2. Each chapter summary will be embedded for semantic search / retrieval.

For each heading, set TWO orthogonal fields:

• is_content — true for substantive content headings; false for non-content sections
  (Abstract, References, Bibliography, Acknowledgements, Author Contributions,
  Funding, Conflicts of Interest, Abbreviations, Appendix, Supplementary Material).
  Non-content sections are excluded from summarization entirely.

• chunk_boundary — true if this heading starts a new chunk for summarization.
  Only meaningful when is_content=true. Non-content headings should always have
  chunk_boundary=false.

HOW TO REASON ABOUT SIZE:
Each heading shows its section's token count. When you set chunk_boundary=false on a
content heading, its tokens merge into the preceding chunk. Before setting
chunk_boundary=true, mentally sum the tokens of consecutive sections that would form
the resulting chunk. Aim for every chunk to be at least 3 000 tokens.

RULES:
1. The first content heading in the document MUST have chunk_boundary=true.
   Every document should have at least 2 content chunks unless the entire document
   is under 6 000 tokens.
2. Most documents have a natural three-part structure: introductory material, main
   content, and discussion/conclusion. Prefer preserving these as separate chunks
   even if one part is slightly under 3 000 tokens.
3. If a content section is under 3 000 tokens, default to chunk_boundary=false so it
   merges with its neighbour. The ONLY exception is if the section is on a completely
   unrelated topic (e.g. "Methods" vs "Results") where combining would make the
   embedding meaningless. Sub-topics within the same broad subject do NOT qualify —
   they belong together.
4. Very large sections (>60 000 tokens) should be split at sub-headings that represent
   genuine topic shifts.
5. Top-level headings (chapters, parts, numbered sections like 1, 2, 3) are natural
   chunk boundaries — but still merge them if the result would be under 3 000 tokens
   and they cover the same broad topic as their neighbour.
6. Sub-sections (1.1, 1.2) are almost never chunk boundaries.
7. Non-content sections must ALWAYS have is_content=false and chunk_boundary=false.

When in doubt, prefer fewer, larger chunks over many small ones."""

        if is_multi_author:
            system_prompt += (
                "\n\nThis is a multi-author book. For each chapter, identify the author name if present in the heading."
            )

        try:
            # Use structured extraction for guaranteed valid JSON
            # No max_tokens limit - large docs can have many headings
            # use_json_schema_method=True for stricter validation, combined with
            # field validator to handle edge case of JSON strings in list fields
            result = await invoke(
                tier=ModelTier.DEEPSEEK_R1,
                system=system_prompt,
                user=heading_list,
                schema=HeadingAnalysisResult,
                config=InvokeConfig(max_tokens=8192),
            )
            analysis = [h.model_dump() for h in result.headings]

            # Build chapter boundaries
            chapters = _build_chapter_boundaries(markdown, headings, analysis)

            # If no chapters identified, try using top-level headings as fallback
            if not chapters:
                logger.info("No chapters identified by LLM, trying top-level heading fallback")
                chapters = create_heading_based_chapters(markdown, headings, ChapterInfo)

                if chapters:
                    logger.info(f"Created {len(chapters)} chapters from top-level headings")
                    # Filter out non-content chapters using heuristic
                    content_chapters = [c for c in chapters if not _is_non_content_heading(c["title"])]
                    excluded_count = len(chapters) - len(content_chapters)
                    if excluded_count:
                        logger.info(f"Filtered {excluded_count} non-content chapters from heading fallback")
                    return {
                        "chapters": content_chapters,
                        "needs_tenth_summary": True,
                        "current_status": "chapters_detected_heading_fallback",
                    }

                # Final fallback: arbitrary chunking
                logger.info("No usable heading structure, using size-based chunking")
                chapters = create_fallback_chunks(markdown, word_count, ChapterInfo)
                return {
                    "chapters": chapters,
                    "needs_tenth_summary": True,
                    "current_status": "chapters_detected_fallback",
                }

            logger.info(f"Detected {len(chapters)} chapters for 10:1 summary")

            return {
                "chapters": chapters,
                "needs_tenth_summary": True,
                "current_status": "chapters_detected",
            }

        except Exception as e:
            # Graceful fallback: chunk document into ~30k word sections
            logger.warning(f"Chapter detection via LLM failed: {e}. Using fallback chunking.")
            chapters = create_fallback_chunks(markdown, word_count, ChapterInfo)

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
