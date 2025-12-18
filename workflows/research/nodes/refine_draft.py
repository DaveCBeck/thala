"""
Refine draft node.

Refines the draft report based on new research findings.
Called after researcher agents return with findings.
"""

import logging
from datetime import datetime
from typing import Any

from workflows.research.state import DeepResearchState, DraftReport
from workflows.research.prompts import REFINE_DRAFT_SYSTEM, get_today_str
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


def _format_findings_for_draft(findings: list) -> str:
    """Format findings for inclusion in draft refinement."""
    if not findings:
        return "No findings available."

    formatted = []
    for f in findings:
        formatted.append(f"""
### {f.get('question_id', 'Unknown Question')}

{f.get('finding', 'No finding')}

**Sources:**
""")
        for s in f.get('sources', []):
            formatted.append(f"- [{s.get('title', 'Untitled')}]({s.get('url', '')})")

        formatted.append(f"\n**Confidence:** {f.get('confidence', 0):.1f}")

        if f.get('gaps'):
            formatted.append(f"\n**Gaps:** {', '.join(f['gaps'])}")

    return "\n".join(formatted)


async def refine_draft(state: DeepResearchState) -> dict[str, Any]:
    """Refine the draft report with new research findings.

    Takes the current draft and new findings, produces an updated draft.

    Returns:
        - draft_report: Updated DraftReport
        - current_status: updated status
    """
    brief = state.get("research_brief", {})
    findings = state.get("research_findings", [])
    current_draft = state.get("draft_report")

    draft_content = current_draft["content"] if current_draft else ""
    version = (current_draft.get("version", 0) if current_draft else 0) + 1

    # Format new findings
    new_findings_text = _format_findings_for_draft(findings)

    prompt = REFINE_DRAFT_SYSTEM.format(
        date=get_today_str(),
        research_brief=brief.get("topic", "Unknown topic"),
        draft_report=draft_content or "No draft yet - create initial draft.",
        new_findings=new_findings_text,
    )

    llm = get_llm(ModelTier.SONNET)  # Sonnet for draft updates

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        updated_content = response.content.strip()

        # Estimate remaining gaps from findings
        all_gaps = []
        for f in findings:
            all_gaps.extend(f.get("gaps", []))
        unique_gaps = list(set(all_gaps))

        new_draft = DraftReport(
            content=updated_content,
            version=version,
            last_updated=datetime.utcnow(),
            gaps_remaining=unique_gaps,
        )

        logger.info(f"Refined draft to version {version}, {len(unique_gaps)} gaps remaining")

        return {
            "draft_report": new_draft,
            "current_status": "supervising",
        }

    except Exception as e:
        logger.error(f"Refine draft failed: {e}")

        # Keep current draft on error
        return {
            "errors": [{"node": "refine_draft", "error": str(e)}],
            "current_status": "supervising",
        }
