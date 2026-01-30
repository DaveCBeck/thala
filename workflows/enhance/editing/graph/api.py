"""Main entry point for the editing workflow."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from langsmith import traceable

from workflows.shared.quality_config import QualityTier

from ..quality_presets import EDITING_QUALITY_PRESETS
from ..state import build_initial_state
from .construction import editing_graph

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EditingWorkflow")
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

    # Generate run ID
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
        run_id = uuid.UUID(langsmith_run_id)
        result = await editing_graph.ainvoke(
            initial_state,
            config={
                "run_id": run_id,
                "run_name": f"editing:{topic[:30]}",
                "recursion_limit": 100,  # Higher limit for many parallel sections
                "tags": [
                    f"quality:{quality}",
                    "workflow:editing",
                ],
                "metadata": {
                    "topic": topic[:100],
                    "quality_tier": quality,
                    "doc_length": len(document),
                },
            },
        )

        final_document = result.get("final_document", "")
        errors = result.get("errors", [])
        status = result.get("status", "failed")

        # Override status based on output
        if not final_document:
            status = "failed"
        elif errors and status == "success":
            status = "partial"

        # Count sections from V2 state
        sections = result.get("sections", [])
        source_count = len(sections)

        # Track sections modified during V2 structure phase
        rewritten_sections = result.get("rewritten_sections", [])
        sections_modified = len(rewritten_sections)

        logger.info(
            f"Editing workflow completed: status={status}, "
            f"sections={source_count}, modified={sections_modified}, errors={len(errors)}"
        )

        return {
            "final_report": final_document,
            "status": status,
            "langsmith_run_id": langsmith_run_id,
            "errors": errors,
            "source_count": source_count,
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at", datetime.now(timezone.utc)),
            "changes_summary": result.get("changes_summary", ""),
            # V2 verification
            "verification": result.get("verification", {}),
            # Section stats
            "sections_modified": sections_modified,
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
            "completed_at": datetime.now(timezone.utc),
            "changes_summary": f"Workflow failed: {e}",
        }
