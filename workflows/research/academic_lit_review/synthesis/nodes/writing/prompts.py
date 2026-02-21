"""Prompt templates for writing nodes.

Section word targets are calculated as proportions of the total target word count:
- Introduction: 8%
- Methodology: 6%
- Thematic sections combined: 70% (divided by theme count)
- Discussion: 9%
- Conclusions: 5%
- Abstract: 2% (handled in integration)
"""

# Section proportions of total word count
SECTION_PROPORTIONS = {
    "introduction": 0.08,
    "methodology": 0.06,
    "thematic_total": 0.70,
    "discussion": 0.09,
    "conclusions": 0.05,
    "abstract": 0.02,
}

# Default total word count if not specified
DEFAULT_TARGET_WORDS = 12000


def _word_range(target: int, variance: float = 0.15) -> str:
    """Format a word count target as a range (e.g., '850-1050')."""
    low = int(target * (1 - variance))
    high = int(target * (1 + variance))
    return f"{low}-{high}"


def get_section_targets(total_words: int, theme_count: int = 4) -> dict[str, int]:
    """Calculate word targets for each section based on total target.

    Args:
        total_words: Total target word count for the entire review
        theme_count: Number of thematic sections (affects per-theme target)

    Returns:
        Dictionary mapping section names to their word count targets
    """
    theme_count = max(1, theme_count)  # Avoid division by zero
    thematic_per_section = int(total_words * SECTION_PROPORTIONS["thematic_total"] / theme_count)

    return {
        "introduction": int(total_words * SECTION_PROPORTIONS["introduction"]),
        "methodology": int(total_words * SECTION_PROPORTIONS["methodology"]),
        "thematic_section": thematic_per_section,
        "discussion": int(total_words * SECTION_PROPORTIONS["discussion"]),
        "conclusions": int(total_words * SECTION_PROPORTIONS["conclusions"]),
        "abstract": int(total_words * SECTION_PROPORTIONS["abstract"]),
    }


def get_introduction_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate introduction system prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["introduction"])
    return f"""You are an academic writer drafting the introduction for a systematic literature review.

Write a compelling introduction that:
1. Establishes the importance of the research topic
2. Provides background context
3. States the research questions being addressed
4. Outlines the scope and boundaries of the review
5. Previews the thematic structure
6. Conveys where the field stands right now — what makes this moment in the literature distinctive or consequential

Target length: {_word_range(word_target)} words
Style: Academic, third-person, objective tone
Include: Brief preview of each major theme that will be covered
Framing: Orient the reader to the current state of understanding. The introduction should make clear why a review at this point in time is warranted — what has recently shifted, emerged, or been called into question.

Do NOT include citations in the introduction - it should frame the review.
Do NOT include markdown headings — your output is section body text only. Use `###` for any sub-structure."""


INTRODUCTION_USER_TEMPLATE = """Write an introduction for a literature review on:

Topic: {topic}

Research Questions:
{research_questions}

Thematic Structure (the themes that will be covered):
{themes_overview}

Number of papers reviewed: {paper_count}
Date range of literature: {date_range}"""


def get_methodology_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate methodology system prompt with anti-hallucination constraints."""
    word_target = int(target_words * SECTION_PROPORTIONS["methodology"])
    return f"""You are documenting the methodology for a systematic literature review.

Write a methodology section using ONLY the factual data provided in the user message.

STRICT CONSTRAINTS:
- Never mention databases that are not listed in the data (no Web of Science, Scopus, PubMed, Google Scholar unless explicitly provided)
- Never invent Boolean search queries, supplementary searches, or screening processes not described in the data
- Never claim PRISMA compliance or any framework compliance unless explicitly stated
- Every number you write must come directly from the provided data — do not estimate, round, or extrapolate
- If a pipeline stage is missing from the data, omit it — do not fabricate what happened

PROSE QUALITY:
- Write fluent, readable academic prose — not a mechanical enumeration of pipeline statistics
- Foreground the search logic and rationale; weave numbers in naturally as supporting detail
- You do not need to include every number from the data — omit figures that add no analytical value
- Vary sentence structure; avoid listing every query or parameter in a single run-on sentence
- The methodology should read as though written by the paper's author, not generated from a template

Target length: {_word_range(word_target)} words
Style: Precise, process-honest, AI-neutral academic tone
Structure: Search strategy, selection and filtering, processing, thematic organisation
Do NOT include top-level markdown headings (`#` or `##`) — your output is section body text only. Use `###` for any sub-structure."""


METHODOLOGY_USER_TEMPLATE = """<instructions>
Write a methodology section for the literature review on the topic below.
Use ONLY the data provided in the <transparency_data> section.
Convert the structured data into fluent academic prose.
Do not add information beyond what is provided.
</instructions>

<transparency_data>
TOPIC: {topic}

SOURCE DATABASE: OpenAlex

SEARCH QUERIES USED:
{search_queries_formatted}

DISCOVERY:
- Keyword search results: {keyword_paper_count} papers (from {raw_results_count} candidates, filtered by relevance scoring with threshold >= {relevance_threshold})
- Citation network expansion: {citation_paper_count} papers
{expert_papers_line}
CITATION EXPANSION STAGES:
{diffusion_stages_formatted}
Termination reason: {saturation_reason_formatted}

QUALITY FILTERS APPLIED:
- Minimum citations for older papers: {min_citations_filter}
- Recency window: {recency_years} years
- Recency quota: {recency_quota_pct}%
- Relevance threshold: {relevance_threshold}

PROCESSING OUTCOMES:
- Full-text analysis: {full_text_count} papers
- Metadata-only analysis: {metadata_only_count} papers (analysed from abstracts and OpenAlex metadata; full text was not retrievable)
- Failed retrieval: {papers_failed_count} papers
{fallback_note}
THEMATIC ORGANISATION:
- Method: {clustering_method}
- Clusters: {cluster_count}
- Rationale: {clustering_rationale}

CORPUS:
- Date range: {date_range}
- Final size: {total_corpus_size} papers
</transparency_data>"""


def get_thematic_section_system_prompt(
    target_words: int = DEFAULT_TARGET_WORDS,
    theme_count: int = 4,
) -> str:
    """Generate thematic section system prompt with appropriate word target.

    Args:
        target_words: Total target word count for the entire review
        theme_count: Number of thematic sections to divide the thematic budget across
    """
    theme_count = max(1, theme_count)
    word_target = int(target_words * SECTION_PROPORTIONS["thematic_total"] / theme_count)
    return f"""You are writing a thematic section for an academic literature review.

Guidelines:
1. Start with an overview paragraph introducing the theme
2. Trace how understanding has evolved: what early work established, how subsequent studies complicated or refined the picture, and what the most recent work (2025-2026) has changed or revealed
3. Compare and contrast findings across papers
4. Note agreements, disagreements, and debates — especially where recent evidence has shifted the consensus
5. Identify gaps and limitations
6. Use inline citations: [@CITATION_KEY] format

TEMPORAL NARRATIVE:
- Organise with a temporal spine: signal when findings emerged and how later work built on, revised, or overturned earlier conclusions
- Use temporal markers naturally: "early work suggested…", "by the early 2020s…", "more recent evidence indicates…"
- Foreground 2025-2026 publications — these represent the current frontier and should anchor the section's conclusions where available
- Do not relegate recent work to a "recent developments" paragraph at the end; weave it throughout as the evolving thread of the narrative

Target length: {_word_range(word_target)} words
Style: Academic, analytical, synthesizing (not just summarizing)

IMPORTANT CITATION FORMAT:
- Use [@KEY] where KEY is the Zotero citation key provided
- Example: "Recent studies [@ABC123] have shown..."
- For multiple citations: "Several authors [@ABC123; @DEF456] argue..."

Every factual claim must have a citation. Do not make claims without support.

HEADING FORMAT: Do NOT use `#` or `##` headings — the section header is added automatically.
Use `###` for sub-sections within your theme."""


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


def get_discussion_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate discussion system prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["discussion"])
    return f"""You are writing the discussion section for a systematic literature review.

The discussion should:
1. Synthesize findings ACROSS all themes (not repeat them)
2. Anchor in the current state of the field — what do we now understand that we didn't 2-3 years ago?
3. Identify where recent work (2025-2026) has shifted the consensus, opened new questions, or closed old ones
4. Discuss implications for theory and practice
5. Acknowledge limitations of this review
6. Suggest future research directions grounded in the trajectory of recent findings

Target length: {_word_range(word_target)} words
Style: Analytical, forward-looking, anchored in the present moment
Focus: Integration and implications, NOT summary. The reader should leave with a clear sense of where understanding stands right now and where it is heading.
Do NOT include `#` or `##` headings — the section header is added automatically. Use `###` for any sub-structure."""


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


def get_conclusions_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate conclusions system prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["conclusions"])
    return f"""You are writing the conclusions for a systematic literature review.

The conclusions should:
1. Directly answer each research question
2. Summarize key contributions of the review
3. State the most important takeaways
4. End with implications or call to action

Target length: {_word_range(word_target)} words
Style: Clear, definitive, impactful
Avoid: Introducing new information or hedging excessively
Do NOT include `#` or `##` headings — the section header is added automatically. Use `###` for any sub-structure."""


CONCLUSIONS_USER_TEMPLATE = """Write conclusions for this literature review:

Research Questions:
{research_questions}

Key Findings per Question:
{findings_per_question}

Main Contributions:
{main_contributions}

Write clear, actionable conclusions."""


# Backwards compatibility: static prompts using default target
INTRODUCTION_SYSTEM_PROMPT = get_introduction_system_prompt()
METHODOLOGY_SYSTEM_PROMPT = get_methodology_system_prompt()
THEMATIC_SECTION_SYSTEM_PROMPT = get_thematic_section_system_prompt()
DISCUSSION_SYSTEM_PROMPT = get_discussion_system_prompt()
CONCLUSIONS_SYSTEM_PROMPT = get_conclusions_system_prompt()
