"""
Intelligent chunking for markdown input (skips Marker).
"""

from pathlib import Path

from workflows.shared.text_utils import chunk_by_headings, count_words, estimate_pages


async def smart_chunker(state: dict) -> dict:
    """
    Intelligent chunking for markdown input (skips Marker).

    Reads from resolved_path or input.source and chunks by headings.
    """
    resolved_path = state.get("resolved_path")

    # Read markdown content
    if resolved_path and Path(resolved_path).exists():
        with open(resolved_path, "r", encoding="utf-8") as f:
            markdown = f.read()
    else:
        # Fall back to source if it's markdown text
        markdown = state["input"]["source"]

    # Chunk by headings
    chunks = chunk_by_headings(markdown, max_chunk_size=2000)

    # Calculate metrics
    word_count = count_words(markdown)
    page_count = estimate_pages(markdown)

    processing_result = {
        "markdown": markdown,
        "chunks": chunks,
        "page_count": page_count,
        "word_count": word_count,
        "ocr_method": "none",  # No OCR for markdown
    }

    return {
        "processing_result": processing_result,
        "needs_tenth_summary": word_count > 2000,
        "current_status": "markdown_chunked",
    }
