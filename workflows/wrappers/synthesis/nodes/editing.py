"""Phase 5: Editing node."""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance.editing import editing

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="SynthesisEditing")
async def run_editing(state: dict) -> dict[str, Any]:
    """Run editing workflow on the synthesized document.

    Uses the enhance.editing workflow to:
    - Fix structural issues
    - Improve transitions
    - Enhance coherence
    - Verify citations
    - Polish prose
    """
    input_data = state.get("input", {})
    final_report = state.get("final_report", "")
    quality = input_data.get("quality", "standard")

    if not final_report:
        logger.warning("No document to edit")
        return {
            "editing_result": None,
            "current_phase": "finalize",
        }

    topic = input_data.get("topic", "")

    logger.info(f"Phase 5: Running editing workflow on {len(final_report)} chars")

    try:
        result = await editing(
            document=final_report,
            topic=topic,
            quality=quality,
        )

        edited_report = result.get("final_report", final_report)

        if result.get("status") == "failed":
            logger.warning(f"Editing failed: {result.get('errors', [])}")
            # Keep original report
            edited_report = final_report

        logger.info(
            f"Phase 5 complete: status={result.get('status')}, "
            f"length_change={len(edited_report) - len(final_report)}"
        )

        return {
            "editing_result": result,
            "final_report": edited_report,
            "current_phase": "finalize",
        }

    except Exception as e:
        logger.error(f"Editing failed with exception: {e}")
        return {
            "editing_result": None,
            "current_phase": "finalize",
            "errors": [{"phase": "editing", "error": str(e)}],
        }
