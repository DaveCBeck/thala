"""
Final report node.

Generates the comprehensive final report with citations.
Uses OPUS for high-quality synthesis with prompt caching for cost efficiency.
"""

import logging
from typing import Any

from workflows.research.web_research.state import DeepResearchState
from workflows.research.web_research.prompts import (
    FINAL_REPORT_SYSTEM_STATIC,
    FINAL_REPORT_USER_TEMPLATE,
    get_today_str,
)
from workflows.research.web_research.utils import load_prompts_with_translation
from workflows.shared.llm_utils import InvokeConfig, ModelTier
from workflows.shared.llm_utils.integration_guard import call_text_with_guards

logger = logging.getLogger(__name__)


def _format_all_findings(findings: list) -> str:
    """Format all research findings for final report."""
    if not findings:
        return "No research findings available."

    formatted = []
    source_index = 1
    sources_map = {}

    for f in findings:
        section = f"""
## {f.get("question_id", "Research Finding")}

{f.get("finding", "No finding")}

**Confidence:** {f.get("confidence", 0):.0%}
"""
        # Track sources
        for s in f.get("sources", []):
            url = s.get("url", "")
            if url and url not in sources_map:
                sources_map[url] = source_index
                source_index += 1

        if f.get("gaps"):
            section += f"\n**Remaining gaps:** {', '.join(f['gaps'])}"

        formatted.append(section)

    # Add sources section
    if sources_map:
        formatted.append("\n---\n## Sources from Research\n")
        for url, idx in sorted(sources_map.items(), key=lambda x: x[1]):
            # Find title for this URL
            title = url
            for f in findings:
                for s in f.get("sources", []):
                    if s.get("url") == url:
                        title = s.get("title", url)
                        break
            formatted.append(f"[{idx}] {title}: {url}")

    return "\n".join(formatted)


async def final_report(state: DeepResearchState) -> dict[str, Any]:
    """Generate the comprehensive final research report.

    Uses OPUS for high-quality synthesis that:
    - Integrates all findings
    - Considers memory context (existing knowledge)
    - Provides insightful analysis
    - Includes proper citations

    Returns:
        - final_report: Complete report string
        - citations: Structured citation list
        - current_status: updated status
    """
    brief = state.get("research_brief", {})
    findings = state.get("research_findings", [])
    draft = state.get("draft_report")
    memory_context = state.get("memory_context", "")
    language_config = state.get("primary_language_config")

    all_findings_text = _format_all_findings(findings)
    draft_content = draft["content"] if draft else "No draft available."

    # Build research brief text
    research_brief_text = (
        f"**Topic:** {brief.get('topic', 'Unknown')}\n\n"
        f"**Objectives:**\n"
        + "\n".join([f"- {o}" for o in brief.get("objectives", [])])
        + "\n\n**Key Questions:**\n"
        + "\n".join([f"- {q}" for q in brief.get("key_questions", [])])
    )

    system_prompt_template, user_prompt_template = await load_prompts_with_translation(
        FINAL_REPORT_SYSTEM_STATIC,
        FINAL_REPORT_USER_TEMPLATE,
        language_config,
        "final_report_system",
        "final_report_user",
    )

    # Static system prompt (cached) - only contains instructions
    system_prompt = system_prompt_template.format(date=get_today_str())

    # Dynamic user content - contains all research data
    user_prompt = user_prompt_template.format(
        research_brief=research_brief_text,
        all_findings=all_findings_text,
        draft_report=draft_content,
        memory_context=memory_context or "No prior knowledge available.",
    )

    # Guarded generation: handles max_tokens continuation + shrinkage
    # retry against the draft. max_tokens bumped to 32k to give room for
    # thinking + regurgitation on long drafts. Errors propagate — we no
    # longer fall back to the unprocessed draft, which masked real
    # integration failures. Last incremental checkpoint is retained.
    report = await call_text_with_guards(
        input_content=draft_content,
        tier=ModelTier.OPUS,
        system=system_prompt,
        user=user_prompt,
        config=InvokeConfig(max_tokens=32000),
        label="web_research_final_report",
    )

    citations = _extract_citations(report, findings)
    logger.info(
        f"Generated final report: {len(report)} chars, {len(citations)} citations "
        f"(draft was {len(draft_content)} chars)"
    )

    return {
        "final_report": report,
        "citations": citations,
        "current_status": "saving_findings",
    }


def _extract_citations(report: str, findings: list) -> list[dict]:
    """Extract structured citations from the report."""
    citations = []
    seen_urls = set()

    # Get all sources from findings
    for f in findings:
        for s in f.get("sources", []):
            url = s.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                citations.append(
                    {
                        "url": url,
                        "title": s.get("title", "Unknown"),
                        "relevance": s.get("relevance", s.get("description", "medium")),
                    }
                )

    return citations
