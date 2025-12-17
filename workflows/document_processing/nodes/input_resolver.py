"""
Resolve input source to a path accessible by Marker.
"""

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import httpx

from workflows.shared.text_utils import count_words


async def resolve_input(state: dict) -> dict:
    """
    Resolve input source to path accessible by Marker.

    Handles URLs (download), local files (copy), and markdown text.
    """
    input_data = state["input"]
    source = input_data["source"]

    marker_input_dir = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))
    marker_input_dir.mkdir(parents=True, exist_ok=True)

    # Determine source type
    parsed_url = urlparse(source)
    if parsed_url.scheme in ("http", "https"):
        source_type = "url"
        is_already_markdown = False

        # Download file
        filename = Path(parsed_url.path).name or "downloaded_file"
        resolved_path = marker_input_dir / filename

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
        shutil.copy2(source_path, resolved_path)
        resolved_path = str(resolved_path)

    else:
        # Assume markdown text
        source_type = "markdown_text"
        is_already_markdown = True

        # Write to file for processing
        filename = input_data.get("title", "input").replace(" ", "_") + ".md"
        resolved_path = marker_input_dir / filename

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(source)

        resolved_path = str(resolved_path)

    return {
        "source_type": source_type,
        "resolved_path": resolved_path,
        "is_already_markdown": is_already_markdown,
        "current_status": "input_resolved",
    }
