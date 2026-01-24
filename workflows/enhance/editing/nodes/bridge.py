"""Bridge node to convert V2 structure phase output to V1 format.

This node bridges the V2 structure phase (markdown output) to the
V1 Enhancement/Polish phases (DocumentModel input).
"""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.parser import parse_markdown_to_model
from workflows.enhance.editing.nodes.detect_citations import extract_citation_keys

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="V2ToV1Bridge")
async def v2_to_v1_bridge_node(state: dict) -> dict[str, Any]:
    """Convert V2 markdown output to V1 DocumentModel format.

    This bridge enables the V2 structure phase to feed into the
    existing V1 Enhancement and Polish phases.

    Args:
        state: Current workflow state with V2 final_document

    Returns:
        State update with:
        - updated_document_model: Parsed DocumentModel
        - has_citations: Whether document has [@KEY] citations
        - citation_keys: List of unique citation keys
        - enhance_iteration: Reset to 0
        - enhance_flagged_sections: Empty list
    """
    final_document = state.get("final_document", "")

    if not final_document:
        # Fall back to original input if V2 produced nothing
        final_document = state.get("input", {}).get("document", "")
        logger.warning("No final_document from V2, using original input")

    # Parse markdown to DocumentModel
    document_model = parse_markdown_to_model(final_document)

    logger.info(
        f"Bridge: Parsed V2 output to DocumentModel "
        f"({document_model.total_words} words, {document_model.section_count} sections)"
    )

    # Detect citations for Enhancement phase routing
    citation_keys = extract_citation_keys(final_document)
    has_citations = len(citation_keys) > 0

    if has_citations:
        logger.info(
            f"Bridge: Detected {len(citation_keys)} citations - Enhancement phase will run"
        )
    else:
        logger.info("Bridge: No citations detected - skipping to Polish phase")

    return {
        "updated_document_model": document_model.to_dict(),
        "has_citations": has_citations,
        "citation_keys": citation_keys,
        "enhance_iteration": 0,
        "enhance_flagged_sections": [],
    }
