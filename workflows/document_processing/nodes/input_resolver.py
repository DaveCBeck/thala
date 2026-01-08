"""
Resolve input source to a path accessible by Marker.
"""

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from workflows.shared.text_utils import count_words

logger = logging.getLogger(__name__)

# Max filename length (leave room for extension and timestamp)
MAX_FILENAME_BASE = 100


def _looks_like_file_path(source: str) -> bool:
    """Check if source looks like a file path (not markdown text)."""
    # Check for common file path patterns
    if source.startswith("/") or source.startswith("./") or source.startswith("../"):
        return True
    # Check for file extensions commonly used
    path = Path(source)
    if path.suffix.lower() in (".pdf", ".epub", ".doc", ".docx", ".txt", ".md", ".markdown"):
        return True
    # Check for Windows-style paths
    if len(source) > 2 and source[1] == ":":
        return True
    return False


async def resolve_input(state: dict) -> dict:
    """
    Resolve input source to path accessible by Marker.

    Handles URLs (download), local files (copy), and markdown text.
    """
    input_data = state["input"]
    source = input_data["source"]

    marker_input_dir = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))
    marker_input_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Resolving input source: {source[:100]}...")
    logger.info(f"MARKER_INPUT_DIR: {marker_input_dir}")

    # Determine source type
    parsed_url = urlparse(source)
    if parsed_url.scheme in ("http", "https"):
        source_type = "url"
        is_already_markdown = False

        # Download file
        filename = Path(parsed_url.path).name or "downloaded_file"
        resolved_path = marker_input_dir / filename

        logger.info(f"Downloading URL to: {resolved_path}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(source)
            response.raise_for_status()

            with open(resolved_path, "wb") as f:
                f.write(response.content)

        resolved_path = str(resolved_path)

    elif Path(source).exists():
        source_type = "local_file"
        source_path = Path(source)

        # Check if already markdown
        is_already_markdown = source_path.suffix.lower() in (".md", ".markdown", ".txt")

        # Copy to marker input dir
        resolved_path = marker_input_dir / source_path.name
        file_size = source_path.stat().st_size
        logger.info(f"Copying local file ({file_size / 1024 / 1024:.1f} MB): {source_path} -> {resolved_path}")
        shutil.copy2(source_path, resolved_path)
        resolved_path = str(resolved_path)

    elif _looks_like_file_path(source):
        # Source looks like a file path but doesn't exist - this is an error
        logger.error(f"Source looks like a file path but does not exist: {source}")
        raise FileNotFoundError(f"Input file not found: {source}")

    else:
        # Assume markdown text
        source_type = "markdown_text"
        is_already_markdown = True

        # Write to file for processing (truncate title to avoid filename length errors)
        title = input_data.get("title", "input")
        # Sanitize: replace spaces and remove problematic chars
        safe_title = re.sub(r"[^\w\s-]", "", title).replace(" ", "_")
        # Truncate and add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(safe_title) > MAX_FILENAME_BASE:
            safe_title = safe_title[:MAX_FILENAME_BASE]
        filename = f"{safe_title}_{timestamp}.md"
        resolved_path = marker_input_dir / filename

        logger.info(f"Writing markdown text ({len(source)} chars) to: {resolved_path}")

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(source)

        resolved_path = str(resolved_path)

    logger.info(f"Input resolved: type={source_type}, path={resolved_path}, is_markdown={is_already_markdown}")

    return {
        "source_type": source_type,
        "resolved_path": resolved_path,
        "is_already_markdown": is_already_markdown,
        "current_status": "input_resolved",
    }
