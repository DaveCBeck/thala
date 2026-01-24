"""
Content-metadata validation node for document processing workflow.

Verifies that document content "vaguely matches" its extracted metadata.
Uses quick heuristics first, then LLM semantic check if needed.
"""

import logging
import re
from typing import Any, Optional

from langsmith import traceable
from pydantic import BaseModel, Field

from workflows.document_processing.prompts import DOCUMENT_ANALYSIS_SYSTEM
from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.shared.text_utils import get_first_n_pages, get_last_n_pages

logger = logging.getLogger(__name__)


class ContentMetadataMatch(BaseModel):
    """Result of content-metadata validation."""

    matches: bool = Field(description="True if content plausibly matches the metadata")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    reasoning: str = Field(description="Brief explanation")
    mismatched_fields: list[str] = Field(default_factory=list)


def _quick_heuristic_check(
    content: str,
    metadata: dict,
) -> tuple[Optional[bool], float, str]:
    """
    Fast heuristic checks before LLM.

    Returns:
        (result, confidence, reasoning) where result is:
        - True: Definitely matches (skip LLM)
        - False: Definitely doesn't match (skip LLM)
        - None: Inconclusive, need LLM check
    """
    content_lower = content.lower()
    checks_passed = 0
    checks_total = 0

    # ISBN exact match (strongest signal)
    isbn = metadata.get("isbn")
    if isbn:
        checks_total += 1
        # Normalize ISBN (remove hyphens)
        isbn_clean = isbn.replace("-", "").replace(" ", "")
        if isbn_clean in content.replace("-", "").replace(" ", ""):
            checks_passed += 1
            # ISBN match is very strong signal
            return (True, 0.95, f"ISBN {isbn} found in content")

    # Author name check (at least one author should appear)
    authors = metadata.get("authors", [])
    if authors:
        checks_total += 1
        for author in authors:
            # Check last name (more reliable than full name)
            parts = author.split()
            if parts:
                last_name = parts[-1].lower()
                if len(last_name) > 2 and last_name in content_lower:
                    checks_passed += 1
                    break

    # Year check
    date = metadata.get("date")
    if date:
        # Extract 4-digit year
        year_match = re.search(r"\b(19|20)\d{2}\b", str(date))
        if year_match:
            checks_total += 1
            year = year_match.group()
            if year in content:
                checks_passed += 1

    # If no checks possible, inconclusive
    if checks_total == 0:
        return (None, 0.5, "No metadata fields available for heuristic check")

    # Strong pass: most checks passed
    if checks_passed >= checks_total * 0.7:
        return (None, 0.7, f"Heuristics suggest match ({checks_passed}/{checks_total})")

    # If ISBN was checked and not found, that's suspicious but not definitive
    return (None, 0.5, f"Heuristics inconclusive ({checks_passed}/{checks_total})")


def _format_metadata_for_prompt(metadata: dict) -> str:
    """Format metadata dict for LLM prompt."""
    lines = []
    if metadata.get("title"):
        lines.append(f"Title: {metadata['title']}")
    if metadata.get("authors"):
        lines.append(f"Authors: {', '.join(metadata['authors'])}")
    if metadata.get("date"):
        lines.append(f"Date: {metadata['date']}")
    if metadata.get("publisher"):
        lines.append(f"Publisher: {metadata['publisher']}")
    if metadata.get("isbn"):
        lines.append(f"ISBN: {metadata['isbn']}")
    return "\n".join(lines) if lines else "(no metadata)"


@traceable(run_type="chain", name="ValidateContentMetadata")
async def validate_content_metadata(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Validate that document content matches extracted metadata.

    Uses quick heuristics first, then LLM semantic check if needed.
    On failure, routes to finalize (non-blocking for batch processing).

    Returns validation_passed, validation_confidence, validation_reasoning,
    and optionally errors list.
    """
    try:
        processing_result = state.get("processing_result")
        if not processing_result:
            logger.warning("No processing_result, skipping validation")
            return {
                "validation_passed": True,  # Don't block on missing data
                "validation_confidence": 0.0,
                "validation_reasoning": "Skipped: no processing result",
                "current_status": "validation_skipped",
            }

        metadata = state.get("metadata_updates", {})
        if not metadata:
            logger.warning("No metadata, skipping validation")
            return {
                "validation_passed": True,
                "validation_confidence": 0.0,
                "validation_reasoning": "Skipped: no metadata extracted",
                "current_status": "validation_skipped",
            }

        markdown = processing_result["markdown"]

        # Extract content (same format as metadata/summary agents for caching)
        first_pages = get_first_n_pages(markdown, 10)
        last_pages = get_last_n_pages(markdown, 10)
        content = f"{first_pages}\n\n--- END OF FRONT MATTER ---\n\n{last_pages}"

        # Quick heuristic check
        heuristic_result, heuristic_conf, heuristic_reason = _quick_heuristic_check(
            content, metadata
        )

        if heuristic_result is True:
            logger.info(f"Validation passed via heuristics: {heuristic_reason}")
            return {
                "validation_passed": True,
                "validation_confidence": heuristic_conf,
                "validation_reasoning": heuristic_reason,
                "current_status": "validation_passed",
            }

        # LLM semantic check - prompt format matches metadata/summary agents
        metadata_summary = _format_metadata_for_prompt(metadata)

        user_prompt = f"""{content}

---
Extracted metadata:
{metadata_summary}

---
Task: Determine if this document content plausibly matches the extracted metadata.

Consider:
- Does the content seem to be about what the title suggests?
- Do any author names appear in the text?
- Is there any evidence this is a different document?

Be LENIENT - metadata extraction is imperfect. Only mark as NOT matching if there's clear evidence of mismatch (e.g., completely different topic, wrong language, obviously different authors)."""

        result = await get_structured_output(
            output_schema=ContentMetadataMatch,
            user_prompt=user_prompt,
            system_prompt=DOCUMENT_ANALYSIS_SYSTEM,
            tier=ModelTier.DEEPSEEK_V3,
            enable_prompt_cache=True,
            max_tokens=500,
        )

        if result.matches:
            logger.info(f"Validation passed: {result.reasoning}")
            return {
                "validation_passed": True,
                "validation_confidence": result.confidence,
                "validation_reasoning": result.reasoning,
                "current_status": "validation_passed",
            }
        else:
            # Validation mismatch is an expected condition - triggers fallback in lit review
            logger.info(f"Validation failed: {result.reasoning}")
            return {
                "validation_passed": False,
                "validation_confidence": result.confidence,
                "validation_reasoning": result.reasoning,
                "current_status": "validation_failed",
                "errors": [
                    {
                        "node": "validate_content_metadata",
                        "error": f"Content-metadata mismatch: {result.reasoning}",
                        "severity": "validation_failure",
                        "mismatched_fields": result.mismatched_fields,
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        # On exception, allow workflow to continue (fail open)
        return {
            "validation_passed": True,
            "validation_confidence": 0.0,
            "validation_reasoning": f"Validation error (proceeding anyway): {e}",
            "current_status": "validation_error",
            "errors": [
                {
                    "node": "validate_content_metadata",
                    "error": str(e),
                }
            ],
        }
