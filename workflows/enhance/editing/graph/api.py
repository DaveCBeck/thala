"""Main entry point for the editing workflow."""

import logging
import uuid
from datetime import datetime
from typing import Any

from workflows.shared.quality_config import QualityTier
from workflows.shared.tracing import (
    workflow_traceable,
    add_trace_metadata,
    merge_trace_config,
)

from ..quality_presets import EDITING_QUALITY_PRESETS
from ..state import EditingState, build_initial_state
from .construction import editing_graph

logger = logging.getLogger(__name__)


@workflow_traceable(name="EnhanceEditing", workflow_type="enhance_editing")
async def editing(
    document: str,
    topic: str,
    quality: QualityTier = "standard",
) -> dict[str, Any]:
    """Edit a document for structural coherence and flow.

    This workflow performs structural editing on documents to improve:
    - Section organization and flow
    - Introduction and conclusion presence
    - Redundancy reduction
    - Transition quality
    - Overall coherence

    Args:
        document: The markdown document to edit
        topic: Topic/context for the document (helps with coherence analysis)
        quality: Quality tier - "test", "quick", "standard", "comprehensive", "high_quality"

    Returns:
        Standard workflow result dict:
        - final_report: The edited document
        - status: "success", "partial", or "failed"
        - langsmith_run_id: Tracing ID
        - errors: Any errors encountered
        - source_count: Number of sections processed
        - started_at: Start timestamp
        - completed_at: End timestamp
        - changes_summary: Summary of edits made

    Example:
        result = await editing(
            document=my_markdown_text,
            topic="Machine learning best practices",
            quality="standard",
        )

        if result["status"] == "success":
            with open("edited_document.md", "w") as f:
                f.write(result["final_report"])
    """
    # Get quality settings
    if quality not in EDITING_QUALITY_PRESETS:
        logger.warning(f"Unknown quality '{quality}', using 'standard'")
        quality = "standard"

    quality_settings = dict(EDITING_QUALITY_PRESETS[quality])

    # Add dynamic trace metadata for LangSmith filtering
    add_trace_metadata({
        "quality_tier": quality,
        "topic": topic[:50],
    })

    # Generate run ID for state tracking
    langsmith_run_id = str(uuid.uuid4())

    # Build initial state
    initial_state = build_initial_state(
        document=document,
        topic=topic,
        quality_settings=quality_settings,
        langsmith_run_id=langsmith_run_id,
    )

    doc_preview = document[:100].replace("\n", " ") + "..." if len(document) > 100 else document
    logger.info(
        f"Starting editing workflow: topic='{topic}' "
        f"(quality={quality}, doc_length={len(document)}, preview='{doc_preview}')"
    )
    logger.debug(f"LangSmith run ID: {langsmith_run_id}")

    try:
        result = await editing_graph.ainvoke(
            initial_state,
            config=merge_trace_config({
                "run_name": f"editing:{topic[:30]}",
                "recursion_limit": 100,  # Higher limit for many parallel sections
            }),
        )

        final_document = result.get("final_document", "")
        errors = result.get("errors", [])
        status = result.get("status", "failed")

        # Override status based on output
        if not final_document:
            status = "failed"
        elif errors and status == "success":
            status = "partial"

        # Count sections processed
        document_model = result.get("document_model", {})
        source_count = document_model.get("section_count", 0) if document_model else 0

        logger.info(
            f"Editing workflow completed: status={status}, "
            f"sections={source_count}, errors={len(errors)}"
        )

        return {
            "final_report": final_document,
            "status": status,
            "langsmith_run_id": langsmith_run_id,
            "errors": errors,
            "source_count": source_count,
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at", datetime.utcnow()),
            "changes_summary": result.get("changes_summary", ""),
            # Additional context
            "structure_iterations": result.get("structure_iteration", 0),
            "final_verification": result.get("final_verification", {}),
        }

    except Exception as e:
        logger.error(f"Editing workflow failed: {e}", exc_info=True)
        return {
            "final_report": "",
            "status": "failed",
            "langsmith_run_id": langsmith_run_id,
            "errors": [{"node": "unknown", "error": str(e)}],
            "source_count": 0,
            "started_at": initial_state["started_at"],
            "completed_at": datetime.utcnow(),
            "changes_summary": f"Workflow failed: {e}",
        }
