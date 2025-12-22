"""
Metadata extraction node for document processing workflow.

Uses prompt caching for 90% cost reduction when processing multiple documents.
"""

import json
import logging
from typing import Any

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, extract_json_cached
from workflows.shared.text_utils import get_first_n_pages, get_last_n_pages

logger = logging.getLogger(__name__)

# Static system prompt (cached) - ~300 tokens, saves 90% on cache hits
METADATA_SYSTEM_PROMPT = """You are a bibliographic metadata extraction specialist. Extract structured metadata from document excerpts.

Look for:
- title: Full document title
- authors: List of author names (can be empty list)
- date: Publication date (any format)
- publisher: Publisher name
- isbn: ISBN if present

Also determine:
- is_multi_author: true if this appears to be a multi-author edited volume (look for "edited by" or chapter authors)
- chapter_authors: dict mapping chapter titles to author names (only for multi-author books)

Use null for missing values, empty list for no authors."""

METADATA_SCHEMA = """{
  "title": "string or null",
  "authors": ["string"],
  "date": "string or null",
  "publisher": "string or null",
  "isbn": "string or null",
  "is_multi_author": "boolean",
  "chapter_authors": {"chapter_title": "author_name"}
}"""


async def check_metadata(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Extract/verify metadata from first and last 10 pages.

    Extracts title, authors, date, publisher, isbn, is_multi_author flag,
    and chapter_authors for multi-author books.

    Returns metadata_updates dict and current_status.
    """
    try:
        processing_result = state.get("processing_result")
        if not processing_result:
            logger.error("No processing_result in state")
            return {
                "errors": [{"node": "metadata_agent", "error": "No processing result"}],
            }

        markdown = processing_result["markdown"]

        # Extract first and last 10 pages for metadata
        first_pages = get_first_n_pages(markdown, 10)
        last_pages = get_last_n_pages(markdown, 10)
        content = f"{first_pages}\n\n--- END OF FRONT MATTER ---\n\n{last_pages}"

        # Extract metadata via LLM with prompt caching
        # System prompt is cached for 90% cost reduction on batch processing
        metadata = await extract_json_cached(
            text=content,
            system_instructions=METADATA_SYSTEM_PROMPT,
            schema_hint=METADATA_SCHEMA,
            tier=ModelTier.SONNET,
        )

        logger.info(f"Extracted metadata: {list(metadata.keys())}")

        # Clean up metadata - remove null values
        cleaned = {k: v for k, v in metadata.items() if v is not None}

        # Don't update current_status here - parallel nodes would conflict
        return {
            "metadata_updates": cleaned,
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse metadata JSON: {e}, returning empty dict")
        return {
            "metadata_updates": {},
            "errors": [{"node": "metadata_agent", "error": f"JSON parse error: {e}"}],
        }
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return {
            "errors": [{"node": "metadata_agent", "error": str(e)}],
        }
