"""
Metadata extraction node for document processing workflow.

Uses structured output with prompt caching for cost optimization.
"""

import logging
from typing import Any, Optional

from langsmith import traceable
from pydantic import BaseModel, Field

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.shared.text_utils import get_first_n_pages, get_last_n_pages

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Extracted document metadata."""

    title: Optional[str] = Field(default=None, description="Full document title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    date: Optional[str] = Field(
        default=None, description="Publication date (any format)"
    )
    publisher: Optional[str] = Field(default=None, description="Publisher name")
    isbn: Optional[str] = Field(default=None, description="ISBN if present")
    is_multi_author: bool = Field(
        default=False, description="True if multi-author edited volume"
    )
    chapter_authors: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of chapter titles to author names (for multi-author books)",
    )


METADATA_SYSTEM_PROMPT = """You are a bibliographic metadata extraction specialist. Extract structured metadata from document excerpts.

Look for:
- title: Full document title
- authors: List of author names (can be empty list)
- date: Publication date (any format)
- publisher: Publisher name
- isbn: ISBN if present

Also determine:
- is_multi_author: true if this appears to be a multi-author edited volume (look for "edited by" or chapter authors)
- chapter_authors: dict mapping chapter titles to author names (only for multi-author books)"""


@traceable(run_type="chain", name="CheckMetadata")
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

        # Extract metadata via structured output
        result = await get_structured_output(
            output_schema=DocumentMetadata,
            user_prompt=f"Extract metadata from this document:\n\n{content}",
            system_prompt=METADATA_SYSTEM_PROMPT,
            tier=ModelTier.SONNET,
            enable_prompt_cache=True,
        )

        metadata = result.model_dump()
        logger.info(f"Extracted metadata: {list(metadata.keys())}")

        # Clean up metadata - remove null values and empty defaults
        cleaned = {
            k: v for k, v in metadata.items() if v is not None and v != [] and v != {}
        }

        # Don't update current_status here - parallel nodes would conflict
        return {
            "metadata_updates": cleaned,
        }

    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return {
            "errors": [{"node": "metadata_agent", "error": str(e)}],
        }
