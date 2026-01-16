"""Detect citations node for editing workflow."""

import logging
import re
from typing import Any

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


async def detect_citations_node(state: dict) -> dict[str, Any]:
    """Detect whether document contains Zotero citations.

    This node scans the document for [@KEY] patterns and sets
    has_citations flag accordingly. The enhance and verify-facts
    phases will only run if citations are detected.

    Args:
        state: Current workflow state

    Returns:
        State update with has_citations and citation_keys
    """
    # Get the current document text
    document_model = state.get("updated_document_model", state.get("document_model", {}))

    # If we have a document model dict, render to markdown
    if isinstance(document_model, dict) and "sections" in document_model:
        from workflows.enhance.editing.document_model import DocumentModel
        doc = DocumentModel.from_dict(document_model)
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
            f"Enhancement and verification phases will run."
        )
        logger.debug(f"Citation keys: {citation_keys[:10]}{'...' if len(citation_keys) > 10 else ''}")
    else:
        logger.info(
            "No Zotero citations detected in document. "
            "Skipping enhancement and verification phases."
        )

    return {
        "has_citations": has_citations,
        "citation_keys": citation_keys,
    }


def route_to_enhance_or_polish(state: dict) -> str:
    """Route based on whether document has citations.

    Returns:
        "enhance" if has_citations=True, otherwise "polish"
    """
    if state.get("has_citations", False):
        return "enhance"
    return "polish"
