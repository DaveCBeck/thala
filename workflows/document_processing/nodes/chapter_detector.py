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
from workflows.shared.llm_utils import get_structured_output, ModelTier
from workflows.shared.markdown_utils import extract_headings
from workflows.shared.text_utils import count_words
from workflows.shared.token_utils import estimate_tokens_fast

logger = logging.getLogger(__name__)


class HeadingAnalysis(BaseModel):
    """Analysis of a single heading."""

    heading: str = Field(description="Exact heading text")
    is_chapter: bool = Field(description="Whether this is a chapter boundary")
    chapter_author: Optional[str] = Field(
        default=None, description="Author name if multi-author book"
    )


class ChapterClassification(BaseModel):
    """Classification of a chapter for content filtering."""

    title: str = Field(description="Exact chapter title")
    action: str = Field(
        description="'include' for main content, 'exclude' for non-content, 'combine_with_previous' to merge with preceding chapter"
    )
    reason: Optional[str] = Field(
        default=None, description="Brief reason if not 'include'"
    )


class ChapterClassificationResult(BaseModel):
    """Result of chapter content classification."""

    chapters: list[ChapterClassification] = Field(
        description="Classification of each chapter"
    )

    @field_validator("chapters", mode="before")
    @classmethod
    def parse_json_string(cls, v: Any) -> list:
        """Handle LLM returning JSON string instead of list."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            raise ValueError(
                f"chapters must be a list, got unparseable string: {v[:100]}..."
            )
        return v if v is not None else []


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


def _build_chapter_boundaries(
    markdown: str, headings: list[dict], analysis: list[dict]
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
        if heading_text in analysis_map and analysis_map[heading_text].get(
            "is_chapter"
        ):
            author = analysis_map[heading_text].get("chapter_author")
            chapter_headings.append(
                {
                    "title": heading_text,
                    "position": heading["position"],
                    "author": author,
                }
            )

    # Build chapter boundaries
    chapters = []
    for i, chapter in enumerate(chapter_headings):
        start = chapter["position"]
        # End is start of next chapter, or end of document
        end = (
            chapter_headings[i + 1]["position"]
            if i + 1 < len(chapter_headings)
            else len(markdown)
        )

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


MIN_CHAPTER_WORDS = 100  # Chapters below this are too short to summarize


async def _filter_content_chapters(
    chapters: list[ChapterInfo],
    short_summary: str | None = None,
) -> tuple[list[ChapterInfo], list[ChapterInfo]]:
    """
    Filter chapters to identify main content vs non-content/subsidiary sections.

    Filtering stages:
    1. LLM classification: Identify non-content, or chapters to combine with previous
    2. Word count filter: Skip any remaining chapters below MIN_CHAPTER_WORDS

    The LLM sees ALL chapters with word counts, allowing it to intelligently combine
    short intro sections with their following content rather than discarding them.

    Example scenario:
        Input chapters:
        - Introduction (80 words)      → combine_with_previous (first, so falls back to include)
        - Background (150 words)       → include
        - Methodology (60 words)       → combine_with_previous → merged into Background
        - Data Collection (400 words)  → combine_with_previous → merged into Background
        - Results (1200 words)         → include

        Output: Background (610 words combined), Results (1200 words)

    The short "Methodology" intro paragraph gets merged with substantive sections
    rather than being discarded by a pre-filter.

    Args:
        chapters: List of detected chapters
        short_summary: Optional 100-word summary for document context

    Returns:
        Tuple of (content_chapters, excluded_chapters)
    """
    if not chapters:
        return [], []

    # Stage 1: Prepare chapter list with word counts for LLM classification
    chapter_list = "\n".join(
        [f"- {c['title']} ({c['word_count']:,} words)" for c in chapters]
    )

    # Add document context if available
    context_section = ""
    if short_summary:
        context_section = f"""DOCUMENT SUMMARY (for context):
{short_summary}

"""

    system_prompt = f"""{context_section}Classify each chapter/section. For each, choose one action:

ACTION: "include" - Main content to summarize
- Topic chapters that discuss the document's subject matter
- Author's Introduction, Methods, Results, Discussion, Conclusion
- Numbered chapters (Chapter 1, Chapter 2, etc.)
- In academic papers: ALL sections discussing findings, methods, analysis
- Word count helps: substantial sections (500+ words) are usually content

ACTION: "exclude" - Non-content to skip entirely
- Abstract (already captured in short summary)
- References, Bibliography, Works Cited
- Index, Glossary, Appendix (unless substantive analysis)
- Acknowledgements, Funding, Conflicts of Interest
- About the Author, Contributors, Dedications
- Title pages, Copyright, Credits

ACTION: "combine_with_previous" - Merge into preceding chapter
- Very short sections (<300 words) that continue the previous topic
- Sub-sections that shouldn't stand alone
- Note: First chapter cannot use this action

IMPORTANT: Topic-specific headings are almost always content!
If a heading relates to the document's subject matter (based on the summary above),
it should be "include" even if it sounds like a generic term.

When in doubt, use "include"."""

    try:
        result = await get_structured_output(
            output_schema=ChapterClassificationResult,
            user_prompt=f"CHAPTERS:\n{chapter_list}",
            system_prompt=system_prompt,
            tier=ModelTier.DEEPSEEK_V3,
            max_tokens=4096,
        )

        # Build lookup map
        classification_map = {c.title: c for c in result.chapters}

        content_chapters = []
        excluded_chapters = []

        for i, chapter in enumerate(chapters):
            classification = classification_map.get(chapter["title"])
            if not classification:
                # Not in response, default to include
                content_chapters.append(chapter)
                continue

            action = classification.action.lower().strip()

            if action == "exclude":
                excluded_chapters.append(chapter)
                logger.debug(
                    f"Excluding chapter '{chapter['title']}': {classification.reason}"
                )
            elif action == "combine_with_previous" and content_chapters:
                # Merge this chapter's content range into the previous one
                prev_chapter = content_chapters[-1]
                combined = ChapterInfo(
                    title=prev_chapter["title"],
                    start_position=prev_chapter["start_position"],
                    end_position=chapter["end_position"],
                    author=prev_chapter["author"],
                    word_count=prev_chapter["word_count"] + chapter["word_count"],
                )
                content_chapters[-1] = combined
                logger.debug(
                    f"Combined '{chapter['title']}' with '{prev_chapter['title']}'"
                )
            else:
                # "include" or fallback
                content_chapters.append(chapter)

        if excluded_chapters:
            excluded_titles = [c["title"] for c in excluded_chapters]
            logger.info(
                f"Filtered {len(excluded_chapters)} non-content chapters: {excluded_titles}"
            )

        # Stage 2: Filter out any remaining chapters too short to summarize
        # (after combining, some may still be below threshold)
        final_content = []
        short_chapters = []
        for chapter in content_chapters:
            if chapter["word_count"] < MIN_CHAPTER_WORDS:
                short_chapters.append(chapter)
                logger.debug(
                    f"Excluding short chapter '{chapter['title']}' "
                    f"({chapter['word_count']} words, below {MIN_CHAPTER_WORDS})"
                )
            else:
                final_content.append(chapter)

        if short_chapters:
            logger.info(
                f"Filtered {len(short_chapters)} chapters below {MIN_CHAPTER_WORDS} words "
                f"(post-combine)"
            )
            excluded_chapters.extend(short_chapters)

        content_chapters = final_content

        # Log token counts for content chapters (estimate ~1.3 tokens per word)
        content_words = sum(c["word_count"] for c in content_chapters)
        content_tokens = int(content_words * 1.3)
        excluded_words = sum(c["word_count"] for c in excluded_chapters)
        excluded_tokens = int(excluded_words * 1.3)
        logger.info(
            f"[TOKEN_TRACKING] Post-filter chapters: {len(content_chapters)} chapters, "
            f"{content_tokens:,} tokens ({content_words:,} words) | "
            f"Excluded: {excluded_tokens:,} tokens ({excluded_words:,} words)"
        )

        return content_chapters, excluded_chapters

    except Exception as e:
        logger.warning(
            f"Chapter content filtering failed: {e}. Applying word count filter only."
        )
        # Fallback: just apply word count filter
        content_chapters = []
        excluded_chapters = []
        for chapter in chapters:
            if chapter["word_count"] < MIN_CHAPTER_WORDS:
                excluded_chapters.append(chapter)
            else:
                content_chapters.append(chapter)

        content_words = sum(c["word_count"] for c in content_chapters)
        content_tokens = int(content_words * 1.3)
        logger.info(
            f"[TOKEN_TRACKING] Post-filter chapters (fallback): "
            f"{len(content_chapters)} chapters, {content_tokens:,} tokens"
        )
        return content_chapters, excluded_chapters


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
                "errors": [
                    {"node": "detect_chapters", "error": "No processing result"}
                ],
            }

        markdown = processing_result["markdown"]
        word_count = processing_result.get("word_count", count_words(markdown))

        # Log document token count for tracking
        doc_tokens = estimate_tokens_fast(markdown, with_safety_margin=False)
        logger.info(
            f"[TOKEN_TRACKING] Document content: {doc_tokens:,} tokens "
            f"({word_count:,} words)"
        )

        # Only run 10:1 summary for documents with substantial content
        if word_count < 3000:
            logger.info(
                f"Document too short ({word_count} words), skipping 10:1 summary"
            )
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

        # Prepare heading list for LLM
        heading_list = "\n".join([f"{'#' * h['level']} {h['text']}" for h in headings])

        # Build prompt
        system_prompt = """Analyze document headings and identify which ones represent major section divisions.
Mark each heading with is_chapter=true if it represents a major division boundary, false otherwise.

Guidelines:
- Major divisions include: chapters, parts, numbered top-level sections (1, 2, 3 or 1.0, 2.0), or consistently-styled major headings
- Be LIBERAL in identifying divisions - if headings appear to mark major content shifts, mark them as chapters
- Common patterns: "Chapter N", "Part N", just numbers (1, 2, 3), Roman numerals (I, II, III), or descriptive titles at consistent heading levels
- For academic papers: Introduction, Methods, Results, Discussion, Conclusion are major divisions
- Sub-sections within chapters (e.g., 1.1, 1.2 under section 1) should be marked false
- When in doubt, prefer marking more headings as chapters rather than fewer - splitting is better than one huge chunk"""

        if is_multi_author:
            system_prompt += "\n\nThis is a multi-author book. For each chapter, identify the author name if present in the heading."

        try:
            # Use structured extraction for guaranteed valid JSON
            # No max_tokens limit - large docs can have many headings
            # use_json_schema_method=True for stricter validation, combined with
            # field validator to handle edge case of JSON strings in list fields
            result = await get_structured_output(
                output_schema=HeadingAnalysisResult,
                user_prompt=heading_list,
                system_prompt=system_prompt,
                tier=ModelTier.DEEPSEEK_V3,
                max_tokens=8192,
            )
            analysis = [h.model_dump() for h in result.headings]

            # Build chapter boundaries
            chapters = _build_chapter_boundaries(markdown, headings, analysis)

            # If no chapters identified, try using top-level headings as fallback
            if not chapters:
                logger.info(
                    "No chapters identified by LLM, trying top-level heading fallback"
                )
                chapters = create_heading_based_chapters(
                    markdown, headings, ChapterInfo
                )

                if chapters:
                    logger.info(
                        f"Created {len(chapters)} chapters from top-level headings"
                    )
                    # Filter to content chapters only
                    short_summary = state.get("short_summary_english") or state.get(
                        "short_summary"
                    )
                    content_chapters, excluded = await _filter_content_chapters(
                        chapters, short_summary=short_summary
                    )
                    logger.info(
                        f"After filtering: {len(content_chapters)} content chapters "
                        f"({len(excluded)} excluded)"
                    )
                    return {
                        "chapters": content_chapters,
                        "needs_tenth_summary": True,
                        "current_status": "chapters_detected_heading_fallback",
                    }

                # Final fallback: arbitrary chunking
                logger.warning("No usable heading structure, using size-based chunking")
                chapters = create_fallback_chunks(markdown, word_count, ChapterInfo)
                return {
                    "chapters": chapters,
                    "needs_tenth_summary": True,
                    "current_status": "chapters_detected_fallback",
                }

            logger.info(f"Detected {len(chapters)} chapters for 10:1 summary")

            # Filter to content chapters only
            short_summary = state.get("short_summary_english") or state.get(
                "short_summary"
            )
            content_chapters, excluded = await _filter_content_chapters(
                chapters, short_summary=short_summary
            )
            logger.info(
                f"After filtering: {len(content_chapters)} content chapters "
                f"({len(excluded)} excluded)"
            )

            return {
                "chapters": content_chapters,
                "needs_tenth_summary": True,
                "current_status": "chapters_detected",
            }

        except Exception as e:
            # Graceful fallback: chunk document into ~30k word sections
            logger.warning(
                f"Chapter detection via LLM failed: {e}. Using fallback chunking."
            )
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
