"""Combine phase: merge academic lit review and web research reports.

Produces a unified report that integrates both academic and web research
findings, using consistent [@ZOTERO_KEY] citation format throughout so
downstream supervision and evening reads phases can resolve all sources.
"""

import logging
import re
from typing import Any, Optional

from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

logger = logging.getLogger(__name__)

COMBINE_SYSTEM_PROMPT = """\
You are a research editor merging an academic literature review with web research findings into a single unified report.

The academic report uses Pandoc-style citations in [@KEY] format. The web research report may use either [@KEY] format or numbered [N] format. Both sets of citations are backed by a reference manager — treat them as equally valid sources. Maintain whichever format each citation originally uses; do NOT convert between formats. If the same paper appears in both reports under different citation keys (e.g., as both [@SOMEKEY] and [N]), keep only the [@KEY] version and drop the [N] duplicate.

Guidelines:
1. Use the academic lit review as the structural backbone — it has thematic clusters and methodology.
2. Integrate web research findings where they add value: recency, breadth, commercial/regulatory context, or complementary evidence. Web research may surface preprints, clinical trial updates, regulatory decisions, and industry developments that academic databases haven't indexed yet — these are valuable contributions, not second-class sources.
3. Preserve ALL citations from BOTH reports exactly as they appear — both [@KEY] and [N] format. Do not modify, drop, or renumber any citation keys. Every citation in both input reports must appear in the output.
4. Keep all citations inline in their original format. Do NOT convert citations to footnotes, URLs, or superscript numbers. Do NOT convert [N] citations to [@KEY] or vice versa.
5. Identify contradictions or updates where findings from one source extend or challenge the other. Let the evidence speak — do not assume one source type is inherently more reliable than the other.
6. Integrate findings naturally — do NOT simply concatenate the two reports.
7. The combined report MUST be longer than the academic report alone — you are adding a second source, not replacing the first. Every academic section should retain its full analytical depth; web findings are added alongside, not in place of, academic arguments. A combined report shorter than the academic report is a failure.
8. Integrate ALL web findings into the academic report's existing section structure. Do NOT create a standalone "Recent Developments" section — this produces bolt-on artifacts. If a web finding does not fit naturally into any existing section, create a new thematic section with a specific title (e.g., "SimCells and the Regulatory Classification Frontier"), not a generic recency bucket.
9. CRITICAL — Discussion and Conclusions preservation: The academic report's Discussion and Conclusions contain cross-theme synthesis and analytical arguments that are NOT recoverable from the thematic sections. You MUST:
   a. Preserve every named analytical argument from the academic Discussion (e.g., barrier classification frameworks, demand-side policy analysis, innovation systems diagnosis, efficacy-in-deployment arguments). These are the most valuable parts of the academic report.
   b. Integrate web findings INTO the academic Discussion's existing argumentative structure — add new subsections or extend existing ones, do not replace the academic Discussion with a web-framed rewrite.
   c. The combined Discussion must contain BOTH the academic analytical arguments AND the web research's updated evidence. If the academic Discussion has 6 subsections and the web adds 3 new themes, the combined Discussion should have ~9 subsections.
   d. The combined Conclusions must preserve the academic report's forward-looking judgements and policy recommendations alongside any web-sourced updates.
10. Preserve epistemic hedges and uncertainty qualifications from BOTH sources. If either report flags a claim as uncertain, contested, lacking independent validation, or based on stale comparisons, that hedge MUST appear in the combined report at the point where the claim is stated. Dropping uncertainty qualifications makes the combined report less honest than its inputs — this is never acceptable. If the web report contains a "Remaining Uncertainties" or equivalent section, integrate those hedges inline at the relevant claims rather than collecting them in a separate section.

Output the merged report as markdown. Do NOT include any JSON wrapper or metadata — output ONLY the report text."""


def _build_section_inventory(report: str) -> str:
    """Extract section headings and word counts from a markdown report."""
    lines = report.split("\n")
    sections: list[tuple[str, int]] = []
    current_heading = None
    current_words = 0

    for line in lines:
        heading_match = re.match(r"^(#{1,3})\s+(.+)", line)
        if heading_match:
            if current_heading:
                sections.append((current_heading, current_words))
            current_heading = heading_match.group(2).strip()
            current_words = 0
        else:
            current_words += len(line.split())

    if current_heading:
        sections.append((current_heading, current_words))

    return "\n".join(f"- {heading}: ~{words} words" for heading, words in sections if words > 50)


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

    # Build section inventory from academic report
    section_inventory = _build_section_inventory(academic_report)

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Research questions:\n{rq_text}"
        f"{landscape_section}\n\n"
        f"Academic report section inventory (preserve ALL sections at similar or greater depth):\n{section_inventory}\n\n"
        f"--- ACADEMIC LITERATURE REVIEW ---\n\n{academic_report}\n\n"
        f"--- WEB RESEARCH FINDINGS ---\n\n{web_report}"
    )

    response = await invoke(
        tier=ModelTier.OPUS,
        system=COMBINE_SYSTEM_PROMPT,
        user=user_prompt,
        config=InvokeConfig(effort="high", max_tokens=32768),
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

    # Merge zotero_keys from both sources into a DOI -> key dict.
    # Academic keys are already DOI-keyed.  Web citation keys have no real
    # DOIs, so we use a synthetic "web:<KEY>" prefix — downstream code does
    # safe `if doi in paper_summaries` checks before access, so these
    # synthetic DOIs simply pass through without causing lookups.
    academic_keys = lit_result.get("zotero_keys") or {}
    if not isinstance(academic_keys, dict):
        # Guard against legacy list format
        academic_keys = {}
    web_keys = (web_result.get("citation_keys") or []) if web_result else []
    merged_keys = {**academic_keys}
    for key in web_keys:
        synthetic_doi = f"web:{key}"
        if synthetic_doi not in merged_keys:
            merged_keys[synthetic_doi] = key

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
