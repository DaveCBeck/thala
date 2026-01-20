"""Detect citations node for fact-check workflow."""

import logging
import re
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.document_model import DocumentModel

logger = logging.getLogger(__name__)

# Zotero citation pattern: [@8ALPHANUMERIC]
ZOTERO_CITATION_PATTERN = re.compile(r'\[@([A-Za-z0-9]{8})\]')


def extract_citation_keys(text: str) -> list[str]:
    """Extract all Zotero citation keys from text.

    Args:
        text: Document text to scan

    Returns:
        List of unique citation keys (without the [@] wrapper)
    """
    matches = ZOTERO_CITATION_PATTERN.findall(text)
    # Return unique keys preserving order
    seen = set()
    unique = []
    for key in matches:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


@traceable(run_type="chain", name="FactCheckDetectCitations")
async def detect_citations_node(state: dict) -> dict[str, Any]:
    """Detect whether document contains Zotero citations.

    If has_citations and citation_keys are already set (from editing workflow),
    skip detection and use the provided values.

    Otherwise, scan the document for [@KEY] patterns.

    Args:
        state: Current workflow state

    Returns:
        State update with has_citations and citation_keys
    """
    # Check if citation info already provided
    if state.get("has_citations") is not None and state.get("citation_keys"):
        logger.debug(
            f"Using pre-provided citation info: {len(state['citation_keys'])} citations"
        )
        return {
            "has_citations": state["has_citations"],
            "citation_keys": state["citation_keys"],
        }

    # Get the current document text
    document_model_dict = state.get("document_model", {})

    if document_model_dict and "sections" in document_model_dict:
        doc = DocumentModel.from_dict(document_model_dict)
        text = doc.to_markdown()
    else:
        # Fall back to original input
        text = state.get("input", {}).get("document", "")

    # Extract citation keys
    citation_keys = extract_citation_keys(text)
    has_citations = len(citation_keys) > 0

    if has_citations:
        logger.info(
            f"Detected {len(citation_keys)} unique citations in document. "
            f"Fact-check and reference-check phases will run."
        )
        logger.debug(f"Citation keys: {citation_keys[:10]}{'...' if len(citation_keys) > 10 else ''}")
    else:
        logger.info(
            "No Zotero citations detected in document. "
            "Skipping fact-check and reference-check phases."
        )

    return {
        "has_citations": has_citations,
        "citation_keys": citation_keys,
    }


def route_citations_or_finalize(state: dict) -> str:
    """Route based on whether document has citations.

    Returns:
        "fact_check_router" if has_citations=True, otherwise "finalize"
    """
    if state.get("has_citations", False):
        return "fact_check_router"
    return "finalize"
