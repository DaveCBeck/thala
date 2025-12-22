"""
Synthesize languages node.

Cross-language synthesis for composite multi-lingual research mode.
Integrates findings from multiple language streams to identify unique insights
and cross-cultural perspectives.
"""

import logging
from typing import Any

from workflows.research.state import DeepResearchState, ResearchFinding
from workflows.shared.llm_utils import get_llm, ModelTier, invoke_with_cache
from workflows.research.config.languages import LANGUAGE_NAMES

logger = logging.getLogger(__name__)


def _format_language_findings(
    language_findings: dict[str, list[ResearchFinding]],
    language_code: str,
) -> str:
    """Format findings from a specific language for synthesis."""
    findings = language_findings.get(language_code, [])
    if not findings:
        return f"No findings from {LANGUAGE_NAMES.get(language_code, language_code)} sources."

    formatted = []
    for f in findings:
        section = f"""
**Question:** {f.get('question_id', 'Unknown')}

{f.get('finding', 'No finding')}

Confidence: {f.get('confidence', 0):.0%}
"""
        if f.get('gaps'):
            section += f"Remaining gaps: {', '.join(f['gaps'])}\n"

        formatted.append(section)

    return "\n".join(formatted)


async def synthesize_languages(state: DeepResearchState) -> dict[str, Any]:
    """Synthesize research findings from multiple language streams.

    Creates cross-language analysis that:
    - Identifies unique insights from each language not available in English
    - Notes consensus across different language sources
    - Highlights cultural/regional differences and disagreements
    - Integrates findings while preserving distinct perspectives

    Returns:
        - language_synthesis: Cross-language synthesis text
        - current_status: updated status
    """
    language_findings = state.get("language_findings", {})
    active_languages = state.get("active_languages", [])
    research_brief = state.get("research_brief", {})

    # Skip if single language or no findings
    if not language_findings or len(language_findings) <= 1:
        logger.info("Single language mode or no language findings - skipping synthesis")
        return {"current_status": "final_report"}

    if not active_languages or len(active_languages) <= 1:
        logger.info("Only one active language - skipping synthesis")
        return {"current_status": "final_report"}

    logger.info(f"Synthesizing findings from {len(active_languages)} languages: {active_languages}")

    # Build findings by language
    findings_by_language = {}
    for lang_code in active_languages:
        language_name = LANGUAGE_NAMES.get(lang_code, lang_code)
        findings_text = _format_language_findings(language_findings, lang_code)
        findings_by_language[language_name] = findings_text

    # Build research context
    research_context = (
        f"**Topic:** {research_brief.get('topic', 'Unknown')}\n\n"
        f"**Objectives:**\n" + "\n".join([f"- {o}" for o in research_brief.get('objectives', [])]) +
        f"\n\n**Key Questions:**\n" + "\n".join([f"- {q}" for q in research_brief.get('key_questions', [])])
    )

    # Static system prompt for synthesis
    system_prompt = """You are a cross-cultural research synthesizer. Your task is to analyze research findings from multiple language sources and create a synthesis that highlights:

1. **Unique Insights**: What did each language source reveal that was NOT available in other languages? These are the most valuable findings - perspectives, data, or cultural context unique to that language's sources.

2. **Cross-Cultural Consensus**: Where do sources from different languages AGREE? This indicates robust findings that transcend cultural boundaries.

3. **Cultural/Regional Differences**: Where do sources DISAGREE or present different perspectives? These differences often reveal important cultural, regional, or methodological variations.

4. **Integrated Analysis**: Synthesize findings into coherent insights that preserve distinct perspectives rather than averaging them out.

Guidelines:
- DO NOT simply translate findings - synthesize them into NEW insights
- Explicitly note which language source provided which unique perspective
- Highlight contradictions and explain possible reasons (cultural, temporal, methodological)
- Preserve nuance - don't force consensus where legitimate differences exist
- Focus on actionable insights from the cross-language comparison

Your synthesis should help researchers understand what they learned BY searching multiple languages, not just what they learned ABOUT the topic."""

    # Build user prompt with all language findings
    language_sections = []
    for language_name, findings_text in findings_by_language.items():
        language_sections.append(f"## Findings from {language_name} Sources\n\n{findings_text}")

    user_prompt = f"""Research Context:
{research_context}

---

{chr(10).join(language_sections)}

---

Create a cross-language synthesis following the guidelines above. Structure your analysis clearly with sections for unique insights, consensus, differences, and integrated conclusions."""

    llm = get_llm(ModelTier.OPUS, max_tokens=8192)

    try:
        response = await invoke_with_cache(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Extract content
        if isinstance(response.content, str):
            synthesis = response.content.strip()
        elif isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if isinstance(first_block, dict):
                synthesis = first_block.get("text", "").strip()
            elif hasattr(first_block, "text"):
                synthesis = first_block.text.strip()
            else:
                synthesis = str(first_block).strip()
        else:
            synthesis = str(response.content).strip()

        logger.info(f"Generated language synthesis: {len(synthesis)} chars")

        return {
            "language_synthesis": synthesis,
            "current_status": "final_report",
        }

    except Exception as e:
        logger.error(f"Language synthesis failed: {e}")

        # Fallback: provide simple concatenation notice
        fallback = f"""# Cross-Language Research Findings

**Note:** Automated synthesis encountered an error. Below are findings organized by language.

"""
        for language_name, findings_text in findings_by_language.items():
            fallback += f"\n## {language_name} Sources\n{findings_text}\n"

        fallback += f"\n**Error:** {str(e)}"

        return {
            "language_synthesis": fallback,
            "errors": [{"node": "synthesize_languages", "error": str(e)}],
            "current_status": "final_report",
        }
