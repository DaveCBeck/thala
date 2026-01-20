"""Finalize node for fact-check workflow."""

import logging
from datetime import datetime
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.document_model import DocumentModel

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="FactCheckFinalize")
async def finalize_node(state: dict) -> dict[str, Any]:
    """Finalize the fact-check workflow and render final document.

    Args:
        state: Current workflow state

    Returns:
        State update with final_document, status, and changes_summary
    """
    # Get the document model (may be updated or original)
    document_model_dict = state.get("updated_document_model", state.get("document_model"))

    if not document_model_dict:
        logger.error("No document model available for finalization")
        return {
            "final_document": "",
            "status": "failed",
            "completed_at": datetime.utcnow(),
            "changes_summary": "No document model available",
        }

    document_model = DocumentModel.from_dict(document_model_dict)

    # Render final document
    final_document = document_model.to_markdown()

    # Check if we actually did any work
    has_citations = state.get("has_citations", False)
    applied_edits = state.get("applied_edits", [])
    skipped_edits = state.get("skipped_edits", [])
    fact_check_results = state.get("fact_check_results", [])
    reference_check_results = state.get("reference_check_results", [])
    unresolved_items = state.get("unresolved_items", [])
    errors = state.get("errors", [])

    # Determine status
    if errors:
        status = "partial"
    elif not has_citations:
        status = "skipped"
    elif applied_edits or fact_check_results or reference_check_results:
        status = "success"
    else:
        status = "success"  # Ran but found nothing to change

    # Build changes summary
    summary_parts = []

    if not has_citations:
        summary_parts.append("No citations detected - fact-check skipped")
    else:
        # Fact-check summary
        total_claims = sum(r.get("claims_checked", 0) for r in fact_check_results)
        if total_claims:
            summary_parts.append(f"Fact-checked {total_claims} claims")

        # Reference-check summary
        total_citations = sum(len(r.get("citations_found", [])) for r in reference_check_results)
        invalid_citations = sum(len(r.get("invalid_citations", [])) for r in reference_check_results)
        if total_citations:
            summary_parts.append(
                f"Validated {total_citations} citations ({invalid_citations} invalid)"
            )

        # Edits summary
        if applied_edits:
            summary_parts.append(f"Applied {len(applied_edits)} corrections")
        if skipped_edits:
            summary_parts.append(f"Skipped {len(skipped_edits)} edits")
        if unresolved_items:
            summary_parts.append(f"{len(unresolved_items)} unresolved issues logged")

    if errors:
        summary_parts.append(f"{len(errors)} errors encountered")

    changes_summary = "; ".join(summary_parts) if summary_parts else "No changes made"

    logger.info(f"Fact-check finalized: status={status}, {changes_summary}")

    return {
        "final_document": final_document,
        "status": status,
        "completed_at": datetime.utcnow(),
        "changes_summary": changes_summary,
    }
