"""
Compress research node.

Compresses raw research findings from researcher agents into structured findings.
This node is typically called after researcher agents complete their work.
"""

import logging
from typing import Any

from workflows.research.web_research.state import DeepResearchState

logger = logging.getLogger(__name__)


async def compress_research(state: DeepResearchState) -> dict[str, Any]:
    """Compress research findings from multiple sources.

    This is a utility node that can be used to re-compress or aggregate
    findings if needed. In most cases, compression happens in the
    researcher subgraph.

    Returns:
        - research_findings: Updated/compressed findings
        - current_status: updated status
    """
    findings = state.get("research_findings", [])

    if not findings:
        logger.debug("No findings to compress")
        return {"current_status": "supervising"}

    # Check if findings need re-compression (e.g., low confidence)
    low_confidence_findings = [f for f in findings if f.get("confidence", 1.0) < 0.5]

    if not low_confidence_findings:
        logger.debug(f"All {len(findings)} findings have adequate confidence")
        return {"current_status": "supervising"}

    logger.debug(
        f"Re-compressing {len(low_confidence_findings)} low-confidence findings"
    )

    # For now, just pass through - re-compression logic can be added later
    return {"current_status": "supervising"}
