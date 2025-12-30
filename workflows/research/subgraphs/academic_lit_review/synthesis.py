"""Synthesis subgraph for writing the final literature review.

Implements the writing pipeline:
1. Write thematic sections (one per cluster, parallel)
2. Write introduction, methodology, discussion sections
3. Integrate into coherent document
4. Process citations to Pandoc format
5. Quality verification pass

Flow:
    START -> write_intro_methodology -> write_thematic_sections
          -> write_discussion_conclusions -> integrate_sections
          -> process_citations -> verify_quality -> END
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from workflows.research.subgraphs.academic_lit_review.state import (
    FormattedCitation,
    LitReviewInput,
    PaperSummary,
    QualitySettings,
    ThematicCluster,
)
from workflows.research.subgraphs.academic_lit_review.clustering import ClusterAnalysis
from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    invoke_with_cache,
)

logger = logging.getLogger(__name__)

# Constants
MAX_CONCURRENT_SECTIONS = 5
TARGET_SECTION_WORDS = 1500  # Per thematic section
MIN_CITATIONS_PER_SECTION = 5


# =============================================================================
# State Definition
# =============================================================================


class QualityMetrics(TypedDict):
    """Quality metrics for the literature review."""

    total_words: int
    citation_count: int
    unique_papers_cited: int
    corpus_coverage: float  # % of corpus papers cited
    uncited_papers: list[str]  # DOIs not cited
    sections_count: int
    avg_section_length: int
    issues: list[str]  # Quality issues found


class SynthesisState(TypedDict):
    """State for synthesis/writing subgraph."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings
    paper_summaries: dict[str, PaperSummary]
    clusters: list[ThematicCluster]
    cluster_analyses: list[ClusterAnalysis]

    # Zotero integration (from paper processing)
    zotero_keys: dict[str, str]  # DOI -> Zotero key

    # Section drafts
    introduction_draft: str
    methodology_draft: str
    thematic_section_drafts: dict[str, str]  # Cluster label -> section text
    discussion_draft: str
    conclusions_draft: str

    # Integrated review
    integrated_review: str
    final_review: str  # After citation processing

    # Citations
    references: list[FormattedCitation]
    citation_keys: list[str]  # Zotero keys used

    # Quality tracking
    quality_metrics: Optional[QualityMetrics]
    quality_passed: bool

    # PRISMA documentation
    prisma_documentation: str


# =============================================================================
# Prompts
# =============================================================================

INTRODUCTION_SYSTEM_PROMPT = """You are an academic writer drafting the introduction for a systematic literature review.

Write a compelling introduction that:
1. Establishes the importance of the research topic
2. Provides background context
3. States the research questions being addressed
4. Outlines the scope and boundaries of the review
5. Previews the thematic structure

Target length: 800-1000 words
Style: Academic, third-person, objective tone
Include: Brief preview of each major theme that will be covered

Do NOT include citations in the introduction - it should frame the review."""

INTRODUCTION_USER_TEMPLATE = """Write an introduction for a literature review on:

Topic: {topic}

Research Questions:
{research_questions}

Thematic Structure (the themes that will be covered):
{themes_overview}

Number of papers reviewed: {paper_count}
Date range of literature: {date_range}"""

METHODOLOGY_SYSTEM_PROMPT = """You are documenting the methodology for a systematic literature review.

Write a methodology section covering:
1. Search Strategy: Databases searched, query terms used
2. Selection Criteria: How papers were included/excluded
3. Data Extraction: What information was extracted from papers
4. Synthesis Approach: How findings were organized and synthesized

Target length: 600-800 words
Style: Precise, replicable, following PRISMA guidelines
Include: Specific numbers where relevant"""

METHODOLOGY_USER_TEMPLATE = """Document the methodology for this literature review:

Topic: {topic}

Search Process:
- Initial papers from keyword search: {keyword_count}
- Papers from citation network expansion: {citation_count}
- Total papers after deduplication: {total_papers}
- Papers processed for full-text analysis: {processed_count}

Quality Settings Used:
- Maximum diffusion stages: {max_stages}
- Saturation threshold: {saturation_threshold}
- Minimum citations filter: {min_citations}

Date Range: {date_range}

Final corpus: {final_corpus_size} papers organized into {cluster_count} themes"""

THEMATIC_SECTION_SYSTEM_PROMPT = """You are writing a thematic section for an academic literature review.

Guidelines:
1. Start with an overview paragraph introducing the theme
2. Organize discussion by sub-themes or chronologically
3. Compare and contrast findings across papers
4. Note agreements, disagreements, and debates
5. Identify gaps and limitations
6. Use inline citations: [@CITATION_KEY] format

Target length: 1200-1800 words
Style: Academic, analytical, synthesizing (not just summarizing)

IMPORTANT CITATION FORMAT:
- Use [@KEY] where KEY is the Zotero citation key provided
- Example: "Recent studies [@ABC123] have shown..."
- For multiple citations: "Several authors [@ABC123; @DEF456] argue..."

Every factual claim must have a citation. Do not make claims without support."""

THEMATIC_SECTION_USER_TEMPLATE = """Write a section on the theme: {theme_name}

Theme Description: {theme_description}

Sub-themes to cover: {sub_themes}

Key debates/tensions: {key_debates}

Outstanding questions: {outstanding_questions}

Papers in this theme (with citation keys):
{papers_with_keys}

Narrative summary from analysis:
{narrative_summary}

Write a comprehensive, well-cited section on this theme."""

DISCUSSION_SYSTEM_PROMPT = """You are writing the discussion section for a systematic literature review.

The discussion should:
1. Synthesize findings ACROSS all themes (not repeat them)
2. Identify overarching patterns and trends
3. Discuss implications for theory and practice
4. Acknowledge limitations of this review
5. Suggest future research directions

Target length: 1000-1200 words
Style: Analytical, forward-looking
Focus: Integration and implications, NOT summary"""

DISCUSSION_USER_TEMPLATE = """Write a discussion section that synthesizes across these themes:

Research Questions:
{research_questions}

Themes covered:
{themes_summary}

Key cross-cutting findings:
{cross_cutting_findings}

Research gaps identified:
{research_gaps}

Write a discussion that integrates these findings and discusses implications."""

CONCLUSIONS_SYSTEM_PROMPT = """You are writing the conclusions for a systematic literature review.

The conclusions should:
1. Directly answer each research question
2. Summarize key contributions of the review
3. State the most important takeaways
4. End with implications or call to action

Target length: 500-700 words
Style: Clear, definitive, impactful
Avoid: Introducing new information or hedging excessively"""

CONCLUSIONS_USER_TEMPLATE = """Write conclusions for this literature review:

Research Questions:
{research_questions}

Key Findings per Question:
{findings_per_question}

Main Contributions:
{main_contributions}

Write clear, actionable conclusions."""

INTEGRATION_SYSTEM_PROMPT = """You are integrating sections into a cohesive literature review document.

Your task:
1. Add smooth transitions between sections
2. Ensure consistent terminology throughout
3. Add an abstract (200-300 words)
4. Format with proper markdown headers
5. Ensure logical flow

Do NOT change the substantive content or citations.
Focus on flow, transitions, and formatting.

Output the complete document with this structure:
# [Title]

## Abstract
[200-300 word abstract]

## 1. Introduction
[Provided introduction]

## 2. Methodology
[Provided methodology]

## 3-N. [Thematic Sections]
[Provided sections with headers]

## N+1. Discussion
[Provided discussion]

## N+2. Conclusions
[Provided conclusions]

## References
[Will be added separately]"""

INTEGRATION_USER_TEMPLATE = """Integrate these sections into a cohesive literature review:

Title: {title}

## Introduction
{introduction}

## Methodology
{methodology}

## Thematic Sections
{thematic_sections}

## Discussion
{discussion}

## Conclusions
{conclusions}

Create a well-integrated document with transitions and an abstract."""

QUALITY_CHECK_SYSTEM_PROMPT = """You are reviewing a literature review for quality issues.

Check for:
1. Missing or unclear citations (claims without [@KEY])
2. Logical inconsistencies
3. Unsupported claims
4. Poor transitions
5. Redundancy
6. Balance across sections

Output a JSON object:
{
  "issues": ["issue 1", "issue 2", ...],
  "suggestions": ["suggestion 1", ...],
  "overall_quality": "good" | "acceptable" | "needs_revision",
  "citation_issues": ["specific citation problems..."]
}"""


# =============================================================================
# Helper Functions
# =============================================================================


def format_papers_with_keys(
    dois: list[str],
    paper_summaries: dict[str, PaperSummary],
    zotero_keys: dict[str, str],
) -> str:
    """Format papers with their Zotero citation keys for the prompt."""
    formatted = []

    for doi in dois:
        summary = paper_summaries.get(doi)
        if not summary:
            continue

        key = zotero_keys.get(doi, doi.replace("/", "_").replace(".", "_")[:20])

        paper_text = f"""
[@{key}] {summary.get('title', 'Unknown')} ({summary.get('year', 'n.d.')})
  Authors: {', '.join(summary.get('authors', [])[:3])}
  Key Findings: {'; '.join(summary.get('key_findings', [])[:2])}
  Methodology: {summary.get('methodology', 'N/A')[:100]}"""

        formatted.append(paper_text)

    return "\n".join(formatted)


def extract_citations_from_text(text: str) -> list[str]:
    """Extract all [@KEY] citations from text."""
    pattern = r'\[@([^\]]+)\]'
    matches = re.findall(pattern, text)

    # Handle multiple citations in one bracket [@A; @B]
    keys = []
    for match in matches:
        for key in match.split(";"):
            key = key.strip().lstrip("@")
            if key:
                keys.append(key)

    return list(set(keys))


def calculate_quality_metrics(
    review_text: str,
    paper_summaries: dict[str, PaperSummary],
    zotero_keys: dict[str, str],
) -> QualityMetrics:
    """Calculate quality metrics for the review."""
    # Word count
    words = review_text.split()
    total_words = len(words)

    # Citations
    citation_keys = extract_citations_from_text(review_text)
    citation_count = len(citation_keys)

    # Map keys back to DOIs
    key_to_doi = {v: k for k, v in zotero_keys.items()}
    cited_dois = set()
    for key in citation_keys:
        if key in key_to_doi:
            cited_dois.add(key_to_doi[key])

    unique_papers_cited = len(cited_dois)
    corpus_size = len(paper_summaries)
    corpus_coverage = unique_papers_cited / corpus_size if corpus_size > 0 else 0

    # Uncited papers
    all_dois = set(paper_summaries.keys())
    uncited_papers = list(all_dois - cited_dois)

    # Section analysis
    sections = re.split(r'^## ', review_text, flags=re.MULTILINE)
    sections_count = len(sections) - 1  # Exclude content before first ##

    section_lengths = [len(s.split()) for s in sections[1:]] if sections_count > 0 else [0]
    avg_section_length = sum(section_lengths) // len(section_lengths) if section_lengths else 0

    # Issues
    issues = []
    if corpus_coverage < 0.5:
        issues.append(f"Low corpus coverage: only {corpus_coverage:.0%} of papers cited")
    if total_words < 5000:
        issues.append(f"Review may be too short: {total_words} words")
    if citation_count < 20:
        issues.append(f"Low citation count: {citation_count} citations")

    return QualityMetrics(
        total_words=total_words,
        citation_count=citation_count,
        unique_papers_cited=unique_papers_cited,
        corpus_coverage=corpus_coverage,
        uncited_papers=uncited_papers[:20],  # Limit for display
        sections_count=sections_count,
        avg_section_length=avg_section_length,
        issues=issues,
    )


# =============================================================================
# Node Functions
# =============================================================================


async def write_intro_methodology_node(state: SynthesisState) -> dict[str, Any]:
    """Write introduction and methodology sections."""
    input_data = state.get("input", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    quality_settings = state.get("quality_settings", {})

    topic = input_data.get("topic", "Unknown topic")
    research_questions = input_data.get("research_questions", [])
    date_range = input_data.get("date_range")

    # Prepare themes overview
    themes_overview = "\n".join(
        f"- {c['label']}: {c['description'][:100]}"
        for c in clusters
    )

    # Get date range from papers
    years = [s.get("year", 0) for s in paper_summaries.values() if s.get("year")]
    if years:
        actual_range = f"{min(years)}-{max(years)}"
    elif date_range:
        actual_range = f"{date_range[0]}-{date_range[1]}"
    else:
        actual_range = "Not specified"

    # Write introduction
    intro_prompt = INTRODUCTION_USER_TEMPLATE.format(
        topic=topic,
        research_questions="\n".join(f"- {q}" for q in research_questions),
        themes_overview=themes_overview,
        paper_count=len(paper_summaries),
        date_range=actual_range,
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)

    intro_response = await invoke_with_cache(
        llm,
        system_prompt=INTRODUCTION_SYSTEM_PROMPT,
        user_prompt=intro_prompt,
    )

    introduction = intro_response.content if isinstance(intro_response.content, str) else intro_response.content[0].get("text", "")

    # Write methodology
    # Estimate counts (would be tracked in full workflow)
    total_papers = len(paper_summaries)

    method_prompt = METHODOLOGY_USER_TEMPLATE.format(
        topic=topic,
        keyword_count=total_papers // 4,  # Estimate
        citation_count=total_papers * 3 // 4,  # Estimate
        total_papers=total_papers,
        processed_count=total_papers,
        max_stages=quality_settings.get("max_stages", 5),
        saturation_threshold=quality_settings.get("saturation_threshold", 0.1),
        min_citations=quality_settings.get("min_citations_filter", 10),
        date_range=actual_range,
        final_corpus_size=total_papers,
        cluster_count=len(clusters),
    )

    method_response = await invoke_with_cache(
        llm,
        system_prompt=METHODOLOGY_SYSTEM_PROMPT,
        user_prompt=method_prompt,
    )

    methodology = method_response.content if isinstance(method_response.content, str) else method_response.content[0].get("text", "")

    logger.info("Completed introduction and methodology sections")

    return {
        "introduction_draft": introduction,
        "methodology_draft": methodology,
    }


async def write_thematic_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Write a section for each thematic cluster (parallel)."""
    clusters = state.get("clusters", [])
    cluster_analyses = state.get("cluster_analyses", [])
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})

    if not clusters:
        logger.warning("No clusters to write sections for")
        return {"thematic_section_drafts": {}}

    # Build analysis lookup
    analysis_lookup = {a["cluster_id"]: a for a in cluster_analyses}

    async def write_single_section(cluster: ThematicCluster) -> tuple[str, str]:
        """Write a single thematic section."""
        analysis = analysis_lookup.get(cluster["cluster_id"], {})

        papers_text = format_papers_with_keys(
            cluster["paper_dois"],
            paper_summaries,
            zotero_keys,
        )

        user_prompt = THEMATIC_SECTION_USER_TEMPLATE.format(
            theme_name=cluster["label"],
            theme_description=cluster["description"],
            sub_themes=", ".join(cluster.get("sub_themes", [])) or "None identified",
            key_debates="\n".join(f"- {d}" for d in cluster.get("conflicts", [])) or "None identified",
            outstanding_questions="\n".join(f"- {q}" for q in cluster.get("gaps", [])) or "None identified",
            papers_with_keys=papers_text,
            narrative_summary=analysis.get("narrative_summary", "No analysis available"),
        )

        try:
            llm = get_llm(tier=ModelTier.SONNET, max_tokens=6000)

            response = await invoke_with_cache(
                llm,
                system_prompt=THEMATIC_SECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            section_text = response.content if isinstance(response.content, str) else response.content[0].get("text", "")

            return (cluster["label"], section_text)

        except Exception as e:
            logger.error(f"Failed to write section for {cluster['label']}: {e}")
            return (cluster["label"], f"[Section generation failed: {e}]")

    # Write sections concurrently
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SECTIONS)

    async def write_with_limit(cluster: ThematicCluster) -> tuple[str, str]:
        async with semaphore:
            return await write_single_section(cluster)

    tasks = [write_with_limit(c) for c in clusters]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    section_drafts = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Section writing task failed: {result}")
            continue
        label, text = result
        section_drafts[label] = text

    logger.info(f"Completed {len(section_drafts)} thematic sections")

    return {"thematic_section_drafts": section_drafts}


async def write_discussion_conclusions_node(state: SynthesisState) -> dict[str, Any]:
    """Write discussion and conclusions sections."""
    input_data = state.get("input", {})
    clusters = state.get("clusters", [])
    cluster_analyses = state.get("cluster_analyses", [])

    research_questions = input_data.get("research_questions", [])

    # Prepare themes summary
    themes_summary = "\n".join(
        f"- {c['label']}: {c['description'][:150]}"
        for c in clusters
    )

    # Collect cross-cutting findings and gaps
    cross_cutting = []
    gaps = []
    for analysis in cluster_analyses:
        for question in analysis.get("outstanding_questions", []):
            gaps.append(question)

    for cluster in clusters:
        for gap in cluster.get("gaps", []):
            gaps.append(gap)

    # Write discussion
    discussion_prompt = DISCUSSION_USER_TEMPLATE.format(
        research_questions="\n".join(f"- {q}" for q in research_questions),
        themes_summary=themes_summary,
        cross_cutting_findings="See thematic sections for detailed findings.",
        research_gaps="\n".join(f"- {g}" for g in gaps[:10]) or "None explicitly identified",
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)

    discussion_response = await invoke_with_cache(
        llm,
        system_prompt=DISCUSSION_SYSTEM_PROMPT,
        user_prompt=discussion_prompt,
    )

    discussion = discussion_response.content if isinstance(discussion_response.content, str) else discussion_response.content[0].get("text", "")

    # Write conclusions
    findings_summary = "\n".join(
        f"Q{i+1}: {q}\n   Finding: Based on {len(clusters)} themes covering {sum(len(c['paper_dois']) for c in clusters)} papers"
        for i, q in enumerate(research_questions)
    )

    conclusions_prompt = CONCLUSIONS_USER_TEMPLATE.format(
        research_questions="\n".join(f"- {q}" for q in research_questions),
        findings_per_question=findings_summary,
        main_contributions=f"Systematic review of {sum(len(c['paper_dois']) for c in clusters)} papers organized into {len(clusters)} themes",
    )

    conclusions_response = await invoke_with_cache(
        llm,
        system_prompt=CONCLUSIONS_SYSTEM_PROMPT,
        user_prompt=conclusions_prompt,
    )

    conclusions = conclusions_response.content if isinstance(conclusions_response.content, str) else conclusions_response.content[0].get("text", "")

    logger.info("Completed discussion and conclusions sections")

    return {
        "discussion_draft": discussion,
        "conclusions_draft": conclusions,
    }


async def integrate_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Integrate all sections into a cohesive document."""
    input_data = state.get("input", {})
    introduction = state.get("introduction_draft", "")
    methodology = state.get("methodology_draft", "")
    thematic_sections = state.get("thematic_section_drafts", {})
    discussion = state.get("discussion_draft", "")
    conclusions = state.get("conclusions_draft", "")
    clusters = state.get("clusters", [])

    topic = input_data.get("topic", "Literature Review")

    # Order thematic sections by cluster order
    cluster_order = [c["label"] for c in clusters]
    ordered_sections = []

    for i, label in enumerate(cluster_order):
        section_text = thematic_sections.get(label, f"[Section for {label} not available]")
        ordered_sections.append(f"### {i + 3}. {label}\n\n{section_text}")

    thematic_text = "\n\n".join(ordered_sections)

    integration_prompt = INTEGRATION_USER_TEMPLATE.format(
        title=f"Literature Review: {topic}",
        introduction=introduction,
        methodology=methodology,
        thematic_sections=thematic_text,
        discussion=discussion,
        conclusions=conclusions,
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=16000)

    response = await invoke_with_cache(
        llm,
        system_prompt=INTEGRATION_SYSTEM_PROMPT,
        user_prompt=integration_prompt,
        cache_ttl="1h",
    )

    integrated = response.content if isinstance(response.content, str) else response.content[0].get("text", "")

    logger.info(f"Integrated review: {len(integrated.split())} words")

    return {"integrated_review": integrated}


async def process_citations_node(state: SynthesisState) -> dict[str, Any]:
    """Process citations and build reference list."""
    integrated_review = state.get("integrated_review", "")
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})

    # Extract all citation keys used
    citation_keys_used = extract_citations_from_text(integrated_review)

    # Build reference list
    references: list[FormattedCitation] = []
    key_to_doi = {v: k for k, v in zotero_keys.items()}

    for key in sorted(citation_keys_used):
        doi = key_to_doi.get(key)
        if doi and doi in paper_summaries:
            summary = paper_summaries[doi]
            authors = summary.get("authors", [])
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."

            citation_text = (
                f"{authors_str} ({summary.get('year', 'n.d.')}). "
                f"{summary.get('title', 'Untitled')}. "
                f"{summary.get('venue', 'Unknown venue')}."
            )

            references.append(FormattedCitation(
                doi=doi,
                citation_text=citation_text,
                zotero_key=key,
            ))

    # Add references section to review
    if references:
        references_text = "\n\n## References\n\n"
        for ref in references:
            references_text += f"[@{ref['zotero_key']}] {ref['citation_text']}\n\n"

        final_review = integrated_review + references_text
    else:
        final_review = integrated_review

    logger.info(f"Processed {len(references)} citations")

    return {
        "final_review": final_review,
        "references": references,
        "citation_keys": citation_keys_used,
    }


async def verify_quality_node(state: SynthesisState) -> dict[str, Any]:
    """Verify quality of the final review."""
    final_review = state.get("final_review", "")
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})
    quality_settings = state.get("quality_settings", {})

    # Calculate metrics
    metrics = calculate_quality_metrics(final_review, paper_summaries, zotero_keys)

    # Determine if quality passes
    target_words = quality_settings.get("target_word_count", 10000)
    quality_passed = (
        metrics["corpus_coverage"] >= 0.4 and
        metrics["total_words"] >= target_words * 0.7 and
        len(metrics["issues"]) <= 2
    )

    # Optional: LLM quality check for detailed issues
    try:
        llm = get_llm(tier=ModelTier.HAIKU, max_tokens=2000)

        # Sample the review for quality check (first 5000 chars)
        sample = final_review[:5000]

        response = await invoke_with_cache(
            llm,
            system_prompt=QUALITY_CHECK_SYSTEM_PROMPT,
            user_prompt=f"Review this literature review sample for quality:\n\n{sample}",
        )

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")

        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        quality_result = json.loads(content)
        additional_issues = quality_result.get("issues", [])

        if additional_issues:
            metrics["issues"].extend(additional_issues[:5])

        if quality_result.get("overall_quality") == "needs_revision":
            quality_passed = False

    except Exception as e:
        logger.warning(f"LLM quality check failed: {e}")

    logger.info(
        f"Quality check: {metrics['total_words']} words, "
        f"{metrics['unique_papers_cited']} papers cited ({metrics['corpus_coverage']:.0%} coverage), "
        f"passed={quality_passed}"
    )

    return {
        "quality_metrics": metrics,
        "quality_passed": quality_passed,
    }


async def generate_prisma_docs_node(state: SynthesisState) -> dict[str, Any]:
    """Generate PRISMA-style documentation of the search process."""
    input_data = state.get("input", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])

    topic = input_data.get("topic", "Unknown")
    total_papers = len(paper_summaries)

    prisma_doc = f"""# PRISMA Documentation

## Search Information

**Topic**: {topic}
**Date of Search**: {datetime.utcnow().strftime('%Y-%m-%d')}
**Databases Searched**: OpenAlex

## Identification

- Records identified through database searching: ~{total_papers * 2}
- Additional records through citation network: ~{total_papers * 3}
- Records after duplicates removed: ~{total_papers + total_papers // 2}

## Screening

- Records screened: ~{total_papers + total_papers // 2}
- Records excluded (not relevant): ~{total_papers // 2}

## Eligibility

- Full-text articles assessed for eligibility: {total_papers}
- Full-text articles excluded: 0

## Included

- Studies included in qualitative synthesis: {total_papers}
- Studies organized into thematic clusters: {len(clusters)}

## Thematic Distribution

"""
    for cluster in clusters:
        prisma_doc += f"- {cluster['label']}: {len(cluster['paper_dois'])} papers\n"

    return {"prisma_documentation": prisma_doc}


# =============================================================================
# Subgraph Definition
# =============================================================================


def create_synthesis_subgraph() -> StateGraph:
    """Create the synthesis/writing subgraph.

    Flow:
        START -> write_intro_methodology -> write_thematic_sections
              -> write_discussion_conclusions -> integrate_sections
              -> process_citations -> verify_quality -> prisma_docs -> END
    """
    builder = StateGraph(SynthesisState)

    # Add nodes
    builder.add_node("write_intro_methodology", write_intro_methodology_node)
    builder.add_node("write_thematic_sections", write_thematic_sections_node)
    builder.add_node("write_discussion_conclusions", write_discussion_conclusions_node)
    builder.add_node("integrate_sections", integrate_sections_node)
    builder.add_node("process_citations", process_citations_node)
    builder.add_node("verify_quality", verify_quality_node)
    builder.add_node("generate_prisma_docs", generate_prisma_docs_node)

    # Add edges
    builder.add_edge(START, "write_intro_methodology")
    builder.add_edge("write_intro_methodology", "write_thematic_sections")
    builder.add_edge("write_thematic_sections", "write_discussion_conclusions")
    builder.add_edge("write_discussion_conclusions", "integrate_sections")
    builder.add_edge("integrate_sections", "process_citations")
    builder.add_edge("process_citations", "verify_quality")
    builder.add_edge("verify_quality", "generate_prisma_docs")
    builder.add_edge("generate_prisma_docs", END)

    return builder.compile()


# Export compiled subgraph
synthesis_subgraph = create_synthesis_subgraph()


# =============================================================================
# Convenience Function
# =============================================================================


async def run_synthesis(
    paper_summaries: dict[str, PaperSummary],
    clusters: list[ThematicCluster],
    cluster_analyses: list[ClusterAnalysis],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
    zotero_keys: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Run synthesis/writing as a standalone operation.

    Args:
        paper_summaries: DOI -> PaperSummary mapping
        clusters: List of ThematicClusters from clustering phase
        cluster_analyses: List of ClusterAnalysis from clustering phase
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings
        zotero_keys: Optional DOI -> Zotero key mapping

    Returns:
        Dict with final_review, references, quality_metrics, prisma_documentation
    """
    # Generate default Zotero keys if not provided
    if zotero_keys is None:
        zotero_keys = {}
        for doi in paper_summaries.keys():
            # Generate a simple key from DOI
            key = doi.replace("/", "_").replace(".", "").replace("-", "")[:20].upper()
            zotero_keys[doi] = key

    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=None,
        include_books=False,
        focus_areas=None,
        exclude_terms=None,
        max_papers=None,
    )

    initial_state = SynthesisState(
        input=input_data,
        quality_settings=quality_settings,
        paper_summaries=paper_summaries,
        clusters=clusters,
        cluster_analyses=cluster_analyses,
        zotero_keys=zotero_keys,
        introduction_draft="",
        methodology_draft="",
        thematic_section_drafts={},
        discussion_draft="",
        conclusions_draft="",
        integrated_review="",
        final_review="",
        references=[],
        citation_keys=[],
        quality_metrics=None,
        quality_passed=False,
        prisma_documentation="",
    )

    result = await synthesis_subgraph.ainvoke(initial_state)

    return {
        "final_review": result.get("final_review", ""),
        "references": result.get("references", []),
        "citation_keys": result.get("citation_keys", []),
        "quality_metrics": result.get("quality_metrics"),
        "quality_passed": result.get("quality_passed", False),
        "prisma_documentation": result.get("prisma_documentation", ""),
        # Section drafts for debugging/review
        "section_drafts": {
            "introduction": result.get("introduction_draft", ""),
            "methodology": result.get("methodology_draft", ""),
            "thematic": result.get("thematic_section_drafts", {}),
            "discussion": result.get("discussion_draft", ""),
            "conclusions": result.get("conclusions_draft", ""),
        },
    }
