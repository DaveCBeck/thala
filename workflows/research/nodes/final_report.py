"""
Final report node.

Generates the comprehensive final report with citations.
Uses OPUS for high-quality synthesis.
"""

import logging
from typing import Any

from workflows.research.state import DeepResearchState
from workflows.research.prompts import FINAL_REPORT_SYSTEM, FINAL_REPORT_HUMAN, get_today_str
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


def _format_all_findings(findings: list) -> str:
    """Format all research findings for final report."""
    if not findings:
        return "No research findings available."

    formatted = []
    source_index = 1
    sources_map = {}  # url -> index

    for f in findings:
        section = f"""
## {f.get('question_id', 'Research Finding')}

{f.get('finding', 'No finding')}

**Confidence:** {f.get('confidence', 0):.0%}
"""
        # Track sources
        for s in f.get('sources', []):
            url = s.get('url', '')
            if url and url not in sources_map:
                sources_map[url] = source_index
                source_index += 1

        if f.get('gaps'):
            section += f"\n**Remaining gaps:** {', '.join(f['gaps'])}"

        formatted.append(section)

    # Add sources section
    if sources_map:
        formatted.append("\n---\n## Sources from Research\n")
        for url, idx in sorted(sources_map.items(), key=lambda x: x[1]):
            # Find title for this URL
            title = url
            for f in findings:
                for s in f.get('sources', []):
                    if s.get('url') == url:
                        title = s.get('title', url)
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

    all_findings_text = _format_all_findings(findings)
    draft_content = draft["content"] if draft else "No draft available."

    prompt = FINAL_REPORT_SYSTEM.format(
        date=get_today_str(),
        research_brief=f"**Topic:** {brief.get('topic', 'Unknown')}\n\n"
                      f"**Objectives:**\n" + "\n".join([f"- {o}" for o in brief.get('objectives', [])]) +
                      f"\n\n**Key Questions:**\n" + "\n".join([f"- {q}" for q in brief.get('key_questions', [])]),
        all_findings=all_findings_text,
        draft_report=draft_content,
        memory_context=memory_context or "No prior knowledge available.",
    )

    llm = get_llm(ModelTier.OPUS, max_tokens=8192)  # OPUS for quality

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": FINAL_REPORT_HUMAN},
        ])

        report = response.content.strip()

        # Extract citations from report
        citations = _extract_citations(report, findings)

        logger.info(f"Generated final report: {len(report)} chars, {len(citations)} citations")

        return {
            "final_report": report,
            "citations": citations,
            "current_status": "saving_findings",
        }

    except Exception as e:
        logger.error(f"Final report generation failed: {e}")

        # Fallback: use draft as report
        fallback_report = f"""# Research Report: {brief.get('topic', 'Unknown Topic')}

## Note
Final report generation encountered an error. Below is the draft report.

{draft_content}

## Error Details
{str(e)}
"""

        return {
            "final_report": fallback_report,
            "citations": [],
            "errors": [{"node": "final_report", "error": str(e)}],
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
                citations.append({
                    "url": url,
                    "title": s.get("title", "Unknown"),
                    "relevance": s.get("relevance", s.get("description", "medium")),
                })

    return citations
