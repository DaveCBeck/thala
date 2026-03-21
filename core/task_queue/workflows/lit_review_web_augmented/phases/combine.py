"""Combine phase: merge academic lit review and web research reports.

Produces a unified report that integrates both academic and web research
findings, using consistent [@ZOTERO_KEY] citation format throughout so
downstream supervision and evening reads phases can resolve all sources.
"""

import logging
from typing import Any, Optional

from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

logger = logging.getLogger(__name__)

COMBINE_SYSTEM_PROMPT = """\
You are a research editor merging an academic literature review with web research findings into a single unified report.

Both reports use Pandoc-style citations in [@KEY] format. Both sets of citations are backed by a reference manager — treat them as equally valid sources.

Guidelines:
1. Use the academic lit review as the structural backbone — it has thematic clusters and methodology.
2. Integrate web research findings where they add value: recency, breadth, commercial/regulatory context, or complementary evidence. Web research may surface preprints, clinical trial updates, regulatory decisions, and industry developments that academic databases haven't indexed yet — these are valuable contributions, not second-class sources.
3. Preserve ALL [@KEY] citations from BOTH reports exactly as they appear. Do not modify, drop, or renumber any citation keys. Every [@KEY] reference in both input reports must appear in the output.
4. When integrating web findings, keep the original [@KEY] citations inline. Do NOT convert them to footnotes, URLs, or superscript numbers.
5. Identify contradictions or updates where findings from one source extend or challenge the other. Let the evidence speak — do not assume one source type is inherently more reliable than the other.
6. Integrate findings naturally — do NOT simply concatenate the two reports.
7. Target similar word count to the academic report (integrate, don't inflate).
8. If web research adds significant recent signal, include a brief "Recent Developments" section or integrate inline depending on volume.

Output the merged report as markdown. Do NOT include any JSON wrapper or metadata — output ONLY the report text."""


async def run_combine_phase(
    lit_result: dict[str, Any],
    web_result: dict[str, Any],
    topic: str,
    augmented_research_questions: list[str],
    recent_landscape: str = "",
) -> dict[str, Any]:
    """Merge academic and web research into a unified report.

    Args:
        lit_result: Academic literature review result (with final_report, paper_corpus, etc.)
        web_result: Web research result (with final_report, citation_keys)
        topic: Research topic
        augmented_research_questions: Research questions (from web scan phase)
        recent_landscape: Brief landscape summary from web scan

    Returns:
        Synthetic lit_result dict containing:
        - final_report: Combined markdown report
        - paper_corpus: Passed through from lit review (unchanged)
        - paper_summaries: Passed through from lit review (unchanged)
        - zotero_keys: Merged from both lit review and web research
        - research_questions: Augmented questions
        - source_breakdown: Counts of academic vs web sources
    """
    academic_report = lit_result.get("final_report", "")
    web_report = web_result.get("final_report", "")
    web_citation_keys = web_result.get("citation_keys", [])

    # Handle single-source graceful degradation
    if not web_report:
        logger.warning("Combine phase: no web report, using academic report only")
        return _build_result(
            lit_result, academic_report, augmented_research_questions, web_result=web_result, web_sources=0
        )

    if not academic_report:
        logger.warning("Combine phase: no academic report, using web report only")
        return _build_result(
            lit_result, web_report, augmented_research_questions, web_result=web_result, academic_sources=0
        )

    # Build context for LLM merge
    rq_text = "\n".join(f"- {q}" for q in augmented_research_questions)
    landscape_section = f"\n\nRecent landscape context:\n{recent_landscape}" if recent_landscape else ""

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Research questions:\n{rq_text}"
        f"{landscape_section}\n\n"
        f"--- ACADEMIC LITERATURE REVIEW ---\n\n{academic_report}\n\n"
        f"--- WEB RESEARCH FINDINGS ---\n\n{web_report}"
    )

    response = await invoke(
        tier=ModelTier.OPUS,
        system=COMBINE_SYSTEM_PROMPT,
        user=user_prompt,
        config=InvokeConfig(effort="high"),
    )

    combined_report = response.content if isinstance(response.content, str) else ""
    # Handle list-type content (extended thinking returns list of blocks)
    if isinstance(response.content, list):
        text_parts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
            elif hasattr(block, "text"):
                text_parts.append(block.text)
        combined_report = "\n".join(text_parts)

    if not combined_report.strip():
        logger.error("Combine phase: LLM returned empty report, falling back to academic report")
        combined_report = academic_report

    academic_source_count = len(lit_result.get("paper_corpus", {}) or {})
    web_source_count = web_result.get("source_count", 0)

    logger.info(
        f"Combined report: {len(combined_report)} chars "
        f"(academic: {academic_source_count} papers, web: {web_source_count} sources, "
        f"web_citation_keys: {len(web_citation_keys)})"
    )

    return _build_result(
        lit_result,
        combined_report,
        augmented_research_questions,
        web_result=web_result,
        academic_sources=academic_source_count,
        web_sources=web_source_count,
    )


def _build_result(
    lit_result: dict[str, Any],
    report: str,
    research_questions: list[str],
    web_result: Optional[dict[str, Any]] = None,
    academic_sources: Optional[int] = None,
    web_sources: Optional[int] = None,
) -> dict[str, Any]:
    """Build the synthetic lit_result dict for downstream phases."""
    if academic_sources is None:
        academic_sources = len(lit_result.get("paper_corpus", {}) or {})

    # Merge zotero_keys from both sources
    academic_keys = lit_result.get("zotero_keys") or []
    web_keys = (web_result.get("citation_keys") or []) if web_result else []
    merged_keys = list(dict.fromkeys(academic_keys + web_keys))  # dedupe, preserve order

    return {
        "final_report": report,
        "paper_corpus": lit_result.get("paper_corpus"),
        "paper_summaries": lit_result.get("paper_summaries"),
        "zotero_keys": merged_keys,
        "research_questions": research_questions,
        "source_breakdown": {
            "academic": academic_sources,
            "web": web_sources or 0,
        },
    }
