"""
Submit document to Marker and wait for completion.
"""

import os
from pathlib import Path

from workflows.shared.marker_client import MarkerClient
from workflows.shared.text_utils import count_words, estimate_pages


async def process_marker(state: dict) -> dict:
    """
    Submit document to Marker and wait for completion.

    Returns processing_result with markdown, chunks, metadata.
    Sets needs_tenth_summary based on word count > 2000.
    """
    input_data = state["input"]
    resolved_path = state["resolved_path"]

    # Get relative path for Marker API
    marker_input_dir = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))
    relative_path = Path(resolved_path).relative_to(marker_input_dir)

    # Submit to Marker
    async with MarkerClient() as client:
        result = await client.convert(
            file_path=str(relative_path),
            quality=input_data.get("quality", "balanced"),
            langs=input_data.get("langs", ["English"]),
        )

    # Calculate metrics
    word_count = count_words(result.markdown)
    page_count = estimate_pages(result.markdown)

    processing_result = {
        "markdown": result.markdown,
        "chunks": result.chunks or [],
        "page_count": page_count,
        "word_count": word_count,
        "ocr_method": result.metadata.get("ocr_method", "unknown"),
    }

    return {
        "processing_result": processing_result,
        "needs_tenth_summary": word_count > 2000,
        "current_status": "marker_processed",
    }
