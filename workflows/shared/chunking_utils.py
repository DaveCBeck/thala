"""Text chunking utilities for document processing."""

import logging

from workflows.shared.markdown_utils import find_word_boundary
from workflows.shared.text_utils import count_words

logger = logging.getLogger(__name__)

# Target chunk size for fallback chunking (in words)
FALLBACK_CHUNK_SIZE = 30000
# Overlap size between chunks (in words) for context continuity
CHUNK_OVERLAP_SIZE = 500


def create_fallback_chunks(markdown: str, word_count: int, chunk_info_class):
    """
    Create pseudo-chapters by splitting document into ~30k word chunks with overlap.

    Used as fallback when heading-based chapter detection fails.
    Splits on paragraph boundaries to avoid breaking mid-sentence.
    Chunks overlap by ~500 words to maintain context continuity.

    Args:
        markdown: Full markdown text
        word_count: Total word count
        chunk_info_class: Class to instantiate for each chunk (e.g., ChapterInfo)

    Returns:
        List of chunk_info_class instances
    """
    num_chunks = max(1, (word_count + FALLBACK_CHUNK_SIZE - 1) // FALLBACK_CHUNK_SIZE)

    # Calculate overlap in characters (approximate)
    avg_chars_per_word = len(markdown) / max(1, word_count)
    overlap_chars = int(CHUNK_OVERLAP_SIZE * avg_chars_per_word)

    # Adjust target size to account for overlap
    target_chunk_size = len(markdown) // num_chunks

    chunks = []
    current_pos = 0
    overlap_start = 0

    for i in range(num_chunks):
        # Start position includes overlap from previous chunk (except first chunk)
        if i == 0:
            start_pos = 0
        else:
            start_pos = overlap_start

        if i == num_chunks - 1:
            # Last chunk gets everything remaining
            end_pos = len(markdown)
        else:
            # Find a good split point near target
            target_pos = current_pos + target_chunk_size
            # Look for paragraph break (double newline) near target
            search_start = max(current_pos, target_pos - 2000)
            search_end = min(len(markdown), target_pos + 2000)
            search_region = markdown[search_start:search_end]

            # Find last paragraph break in search region
            para_break = search_region.rfind("\n\n")
            if para_break != -1:
                end_pos = search_start + para_break + 2
            else:
                # No paragraph break, split at word boundary
                end_pos = find_word_boundary(markdown, target_pos, direction=-1)

        chunk_text = markdown[start_pos:end_pos]
        chunk_word_count = count_words(chunk_text)

        chunks.append(
            chunk_info_class(
                title=f"Section {i + 1}",
                start_position=start_pos,
                end_position=end_pos,
                author=None,
                word_count=chunk_word_count,
            )
        )

        # Set up overlap for next chunk
        current_pos = end_pos
        overlap_start = max(0, end_pos - overlap_chars)
        # Adjust overlap to start at word boundary
        if overlap_start > 0 and i < num_chunks - 1:
            overlap_start = find_word_boundary(markdown, overlap_start, direction=1)

    logger.debug(
        f"Created {len(chunks)} fallback chunks "
        f"(~{FALLBACK_CHUNK_SIZE} words each, {CHUNK_OVERLAP_SIZE} word overlap)"
    )
    return chunks


def create_heading_based_chapters(
    markdown: str,
    headings: list[dict],
    chunk_info_class,
    min_chapters: int = 2,
):
    """
    Create chapters from top-level headings when LLM detection fails.

    Uses the highest heading level (smallest number) that has at least
    min_chapters occurrences. This ensures we get meaningful divisions
    rather than just one or two sections.

    Args:
        markdown: Full markdown text
        headings: List of all headings with positions
        chunk_info_class: Class to instantiate for each chapter (e.g., ChapterInfo)
        min_chapters: Minimum chapters required (default 2)

    Returns:
        List of chunk_info_class instances, or empty list if no suitable structure found
    """
    if not headings:
        return []

    # Group headings by level
    headings_by_level: dict[int, list[dict]] = {}
    for h in headings:
        level = h["level"]
        if level not in headings_by_level:
            headings_by_level[level] = []
        headings_by_level[level].append(h)

    # Find the highest level (lowest number) with enough headings
    for level in sorted(headings_by_level.keys()):
        level_headings = headings_by_level[level]
        if len(level_headings) >= min_chapters:
            logger.debug(
                f"Using H{level} headings as chapter boundaries "
                f"({len(level_headings)} headings)"
            )
            # Build chapters from these headings
            chapters = []
            for i, heading in enumerate(level_headings):
                start = heading["position"]
                # End is start of next heading at same level, or end of document
                if i + 1 < len(level_headings):
                    end = level_headings[i + 1]["position"]
                else:
                    end = len(markdown)

                chapter_text = markdown[start:end]
                chapters.append(
                    chunk_info_class(
                        title=heading["text"],
                        start_position=start,
                        end_position=end,
                        author=None,
                        word_count=count_words(chapter_text),
                    )
                )
            return chapters

    # No level has enough headings
    logger.debug(
        f"No heading level has >= {min_chapters} headings: "
        f"{[(lvl, len(h)) for lvl, h in sorted(headings_by_level.items())]}"
    )
    return []
