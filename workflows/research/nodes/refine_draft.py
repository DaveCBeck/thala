"""
Refine draft node.

Refines the draft report based on new research findings.
Called after researcher agents return with findings.
"""

import logging
from datetime import datetime
from typing import Any

from workflows.research.state import DeepResearchState, DraftReport, calculate_completeness
from workflows.research.prompts import REFINE_DRAFT_SYSTEM, get_today_str
from workflows.research.utils import load_prompts_with_translation
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
    language_config = state.get("primary_language_config")

    draft_content = current_draft["content"] if current_draft else ""
    version = (current_draft.get("version", 0) if current_draft else 0) + 1

    # Format new findings
    new_findings_text = _format_findings_for_draft(findings)

    prompt_template, _ = await load_prompts_with_translation(
        REFINE_DRAFT_SYSTEM,
        "",
        language_config,
        "refine_draft_system",
        "",
    )

    prompt = prompt_template.format(
        date=get_today_str(),
        research_brief=brief.get("topic", "Unknown topic"),
        draft_report=draft_content or "No draft yet - create initial draft.",
        new_findings=new_findings_text,
    )

    llm = get_llm(ModelTier.SONNET)

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

        # Calculate completeness with updated gaps
        diffusion = state.get("diffusion", {})
        new_completeness = calculate_completeness(
            findings=findings,
            key_questions=brief.get("key_questions", []),
            iteration=diffusion.get("iteration", 0),
            max_iterations=diffusion.get("max_iterations", 4),
            gaps_remaining=unique_gaps,
        )

        logger.info(
            f"Refined draft to version {version}, {len(unique_gaps)} gaps remaining, "
            f"completeness: {new_completeness:.0%}"
        )

        return {
            "draft_report": new_draft,
            "diffusion": {
                **diffusion,
                "completeness_score": new_completeness,
            },
            "current_status": "supervising",
        }

    except Exception as e:
        logger.error(f"Refine draft failed: {e}")

        # Keep current draft on error
        return {
            "errors": [{"node": "refine_draft", "error": str(e)}],
            "current_status": "supervising",
        }
