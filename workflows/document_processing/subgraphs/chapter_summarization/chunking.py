"""Chunking strategies for large chapters."""

import logging

logger = logging.getLogger(__name__)

# Maximum content size before chunking (600k chars ≈ 150k tokens, safe for 200k context)
MAX_CHAPTER_CHARS = 600_000
# Target chunk size when splitting large chapters (in characters)
CHUNK_SIZE_CHARS = 500_000
# Overlap between chunks for context continuity
CHUNK_OVERLAP_CHARS = 2000


def chunk_large_content(content: str) -> list[str]:
    """
    Split large content into chunks that fit within token limits.

    Uses paragraph boundaries when possible, with overlap for context continuity.
    """
    if len(content) <= MAX_CHAPTER_CHARS:
        return [content]

    chunks = []
    current_pos = 0

    while current_pos < len(content):
        # Calculate end position for this chunk
        end_pos = min(current_pos + CHUNK_SIZE_CHARS, len(content))

        if end_pos < len(content):
            # Try to find a paragraph break near the target
            search_start = max(current_pos, end_pos - 5000)
            search_region = content[search_start:end_pos]
            para_break = search_region.rfind("\n\n")

            if para_break != -1:
                end_pos = search_start + para_break + 2
            else:
                # Fall back to word boundary
                while end_pos > current_pos and not content[end_pos].isspace():
                    end_pos -= 1

        chunks.append(content[current_pos:end_pos])

        # Move position, accounting for overlap
        if end_pos < len(content):
            current_pos = max(current_pos + 1, end_pos - CHUNK_OVERLAP_CHARS)
            # Adjust to word boundary
            while current_pos < len(content) and not content[current_pos].isspace():
                current_pos += 1
        else:
            break

    logger.info(f"Split large chapter into {len(chunks)} chunks")
    return chunks
