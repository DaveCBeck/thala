"""
Metadata extraction node for document processing workflow.

Uses structured output with prompt caching for cost optimization.
"""

import logging
from typing import Any, Optional

from langsmith import traceable
from pydantic import BaseModel, Field, model_validator

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.metadata_utils import extract_year
from workflows.document_processing.prompts import DOCUMENT_ANALYSIS_SYSTEM
from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.shared.text_utils import get_first_n_pages, get_last_n_pages

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Extracted document metadata with validation."""

    title: Optional[str] = Field(default=None, description="Full document title")
    authors: list[str] = Field(
        default_factory=list,
        description="List of author names in 'First Last' or 'Last, First' format",
    )
    year: Optional[str] = Field(
        default=None, description="Publication year (4-digit, e.g., '2023')"
    )
    date: Optional[str] = Field(
        default=None, description="Full publication date if available (YYYY-MM-DD)"
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

    @model_validator(mode="after")
    def ensure_valid_year(self) -> "DocumentMetadata":
        """Ensure year field has valid value; extract from date if needed."""
        # If year is set, validate it's a proper 4-digit year (1500-2099)
        if self.year:
            extracted = extract_year(self.year)
            self.year = str(extracted) if extracted else None

        # If no year but date is set, extract year from date
        if not self.year and self.date:
            extracted = extract_year(self.date)
            self.year = str(extracted) if extracted else None

        return self


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
        # Content at start for prefix caching (shared with summary_agent)
        user_prompt = f"""{content}

---
Task: Extract bibliographic metadata from the document above.

Extract:
- title: Full document title
- authors: List of author names in 'First Last' format (can be empty list)
- year: Publication year as 4-digit number (e.g., "2023")
- date: Full publication date if available (YYYY-MM-DD format preferred)
- publisher: Publisher name
- isbn: ISBN if present
- is_multi_author: true if multi-author edited volume (look for "edited by" or chapter authors)
- chapter_authors: dict mapping chapter titles to author names (only for multi-author books)"""

        result = await get_structured_output(
            output_schema=DocumentMetadata,
            user_prompt=user_prompt,
            system_prompt=DOCUMENT_ANALYSIS_SYSTEM,
            tier=ModelTier.DEEPSEEK_V3,  # V3 for cost efficiency (R1 also works but costs 2x more)
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
