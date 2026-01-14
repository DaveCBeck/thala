"""Text processing utilities for document workflows."""

import re


def count_words(text: str) -> int:
    """Count words in text (split on whitespace)."""
    return len(text.split())


def estimate_pages(text: str, chars_per_page: int = 3000) -> int:
    """Estimate number of pages based on character count."""
    if chars_per_page <= 0:
        raise ValueError("chars_per_page must be positive")
    return max(1, (len(text) + chars_per_page - 1) // chars_per_page)


def get_first_n_pages(markdown: str, n: int, chars_per_page: int = 3000) -> str:
    """
    Extract first N pages of markdown text.

    Args:
        markdown: Full markdown text
        n: Number of pages to extract
        chars_per_page: Characters per page estimate

    Returns:
        First N pages of text
    """
    if n <= 0:
        return ""
    char_limit = n * chars_per_page
    return markdown[:char_limit]


def get_last_n_pages(markdown: str, n: int, chars_per_page: int = 3000) -> str:
    """
    Extract last N pages of markdown text.

    Args:
        markdown: Full markdown text
        n: Number of pages to extract
        chars_per_page: Characters per page estimate

    Returns:
        Last N pages of text
    """
    if n <= 0:
        return ""
    char_limit = n * chars_per_page
    return markdown[-char_limit:]


def chunk_by_headings(markdown: str, max_chunk_size: int = 2000) -> list[dict]:
    """
    Split markdown into chunks based on heading hierarchy.

    Preserves heading structure in metadata. If chunks exceed max_chunk_size,
    further splits on paragraph boundaries.

    Args:
        markdown: Markdown text to chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of dicts with 'text', 'heading', 'level' keys
    """
    chunks = []

    # Split on markdown headings (1-6 hashes)
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections = []
    last_pos = 0

    for match in heading_pattern.finditer(markdown):
        # Add any text before this heading as a section
        if match.start() > last_pos:
            pre_text = markdown[last_pos : match.start()].strip()
            if pre_text:
                sections.append(
                    {
                        "heading": None,
                        "level": 0,
                        "text": pre_text,
                    }
                )

        # Extract heading info
        level = len(match.group(1))
        heading = match.group(2).strip()

        # Find the end of this section (next heading or end of text)
        next_match = heading_pattern.search(markdown, match.end())
        end_pos = next_match.start() if next_match else len(markdown)

        section_text = markdown[match.end() : end_pos].strip()

        sections.append(
            {
                "heading": heading,
                "level": level,
                "text": section_text,
            }
        )

        last_pos = end_pos

    # If no headings found, treat entire text as one section
    if not sections:
        sections.append(
            {
                "heading": None,
                "level": 0,
                "text": markdown.strip(),
            }
        )

    # Process sections and split large ones
    for section in sections:
        text = section["text"]
        heading = section["heading"]
        level = section["level"]

        # If section fits, add it directly
        if len(text) <= max_chunk_size:
            chunks.append(
                {
                    "text": text,
                    "heading": heading,
                    "level": level,
                }
            )
        else:
            # Split on paragraph boundaries (double newline)
            paragraphs = re.split(r"\n\s*\n", text)
            current_chunk = ""

            for para in paragraphs:
                # If adding this paragraph exceeds limit, save current chunk
                if (
                    current_chunk
                    and len(current_chunk) + len(para) + 2 > max_chunk_size
                ):
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "heading": heading,
                            "level": level,
                        }
                    )
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

            # Add remaining chunk
            if current_chunk:
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "heading": heading,
                        "level": level,
                    }
                )

    return chunks
