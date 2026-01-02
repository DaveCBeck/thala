"""
Submit document to Marker and wait for completion.
"""

import logging
import os
from pathlib import Path

from workflows.shared.marker_client import MarkerClient
from workflows.shared.text_utils import count_words, estimate_pages

logger = logging.getLogger(__name__)


async def process_marker(state: dict) -> dict:
    """
    Submit document to Marker and wait for completion.

    Returns processing_result with markdown, chunks, metadata.
    Sets needs_tenth_summary based on word count > 2000.
    """
    input_data = state["input"]
    resolved_path = state["resolved_path"]

    marker_input_dir = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))

    logger.info(f"Processing with Marker: resolved_path={resolved_path}")
    logger.info(f"MARKER_INPUT_DIR: {marker_input_dir}")

    resolved_path_obj = Path(resolved_path)
    if not resolved_path_obj.exists():
        raise FileNotFoundError(f"Resolved path does not exist: {resolved_path}")

    try:
        relative_path = resolved_path_obj.relative_to(marker_input_dir)
    except ValueError as e:
        logger.error(f"Path mismatch: {resolved_path} is not under {marker_input_dir}")
        raise ValueError(
            f"Path mismatch: resolved_path '{resolved_path}' is not under "
            f"MARKER_INPUT_DIR '{marker_input_dir}'. Check MARKER_INPUT_DIR env var."
        ) from e

    logger.info(f"Submitting to Marker: {relative_path}")

    async with MarkerClient() as client:
        result = await client.convert(
            file_path=str(relative_path),
            quality=input_data.get("quality", "balanced"),
            langs=input_data.get("langs", ["English"]),
            absolute_path=resolved_path,
        )

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
