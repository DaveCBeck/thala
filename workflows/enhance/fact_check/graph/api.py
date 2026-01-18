"""Main entry point for the fact-check workflow."""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from workflows.shared.quality_config import QualityTier

from ..quality_presets import FACT_CHECK_QUALITY_PRESETS
from ..state import build_initial_state
from .construction import fact_check_graph

logger = logging.getLogger(__name__)


async def fact_check(
    document: Optional[str] = None,
    document_model: Optional[dict] = None,
    topic: str = "",
    quality: QualityTier = "standard",
    has_citations: Optional[bool] = None,
    citation_keys: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Fact-check and reference-check a document.

    This workflow verifies factual claims and citations in documents:
    - Pre-screens sections to identify those with verifiable claims
    - Fact-checks claims using paper corpus and Perplexity
    - Validates citation existence and claim support
    - Applies corrections for verified errors

    Args:
        document: The markdown document to check (optional if document_model provided)
        document_model: Pre-parsed document model dict (avoids re-parsing)
        topic: Topic/context for the document (helps with verification)
        quality: Quality tier - "test", "quick", "standard", "comprehensive", "high_quality"
        has_citations: Pre-detected citation flag (from editing workflow)
        citation_keys: Pre-detected citation keys (from editing workflow)

    Returns:
        Standard workflow result dict:
        - final_report: The verified document (with corrections applied)
        - status: "success", "partial", "failed", or "skipped"
        - langsmith_run_id: Tracing ID
        - errors: Any errors encountered
        - started_at: Start timestamp
        - completed_at: End timestamp
        - changes_summary: Summary of verification results
        - fact_check_results: Per-section fact-check results
        - reference_check_results: Per-section reference-check results
        - applied_edits: List of corrections applied
        - unresolved_items: Issues that couldn't be auto-resolved

    Example:
        result = await fact_check(
            document=my_markdown_text,
            topic="Machine learning best practices",
            quality="standard",
        )

        if result["status"] == "success":
            with open("verified_document.md", "w") as f:
                f.write(result["final_report"])
    """
    # Validate input
    if not document and not document_model:
        raise ValueError("Either document or document_model must be provided")

    # Get quality settings
    if quality not in FACT_CHECK_QUALITY_PRESETS:
        logger.warning(f"Unknown quality '{quality}', using 'standard'")
        quality = "standard"

    quality_settings = dict(FACT_CHECK_QUALITY_PRESETS[quality])

    # Generate run ID
    langsmith_run_id = str(uuid.uuid4())

    # Build initial state
    initial_state = build_initial_state(
        document=document,
        document_model=document_model,
        topic=topic,
        has_citations=has_citations,
        citation_keys=citation_keys,
        quality_settings=quality_settings,
        langsmith_run_id=langsmith_run_id,
    )

    if document:
        doc_preview = document[:100].replace("\n", " ") + "..." if len(document) > 100 else document
        logger.info(
            f"Starting fact-check workflow: topic='{topic}' "
            f"(quality={quality}, doc_length={len(document)}, preview='{doc_preview}')"
        )
    else:
        logger.info(
            f"Starting fact-check workflow with pre-parsed model: topic='{topic}' "
            f"(quality={quality}, has_citations={has_citations})"
        )
    logger.debug(f"LangSmith run ID: {langsmith_run_id}")

    try:
        run_id = uuid.UUID(langsmith_run_id)
        result = await fact_check_graph.ainvoke(
            initial_state,
            config={
                "run_id": run_id,
                "run_name": f"fact_check:{topic[:30]}",
                "recursion_limit": 100,  # Higher limit for many parallel sections
            },
        )

        final_document = result.get("final_document", "")
        errors = result.get("errors", [])
        status = result.get("status", "failed")

        # Override status based on output
        if not final_document and document:
            # If we had a document but couldn't produce output, that's a failure
            status = "failed"
        elif errors and status == "success":
            status = "partial"

        logger.info(
            f"Fact-check workflow completed: status={status}, "
            f"errors={len(errors)}"
        )

        return {
            "final_report": final_document,
            "status": status,
            "langsmith_run_id": langsmith_run_id,
            "errors": errors,
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at", datetime.utcnow()),
            "changes_summary": result.get("changes_summary", ""),
            # Detailed results
            "fact_check_results": result.get("fact_check_results", []),
            "reference_check_results": result.get("reference_check_results", []),
            "applied_edits": result.get("applied_edits", []),
            "skipped_edits": result.get("skipped_edits", []),
            "unresolved_items": result.get("unresolved_items", []),
            # Pass through document model for downstream use
            "document_model": result.get("updated_document_model", result.get("document_model")),
            "has_citations": result.get("has_citations", False),
            "citation_keys": result.get("citation_keys", []),
        }

    except Exception as e:
        logger.error(f"Fact-check workflow failed: {e}", exc_info=True)
        return {
            "final_report": document or "",
            "status": "failed",
            "langsmith_run_id": langsmith_run_id,
            "errors": [{"node": "unknown", "error": str(e)}],
            "started_at": initial_state["started_at"],
            "completed_at": datetime.utcnow(),
            "changes_summary": f"Workflow failed: {e}",
            "fact_check_results": [],
            "reference_check_results": [],
            "applied_edits": [],
            "skipped_edits": [],
            "unresolved_items": [],
        }
