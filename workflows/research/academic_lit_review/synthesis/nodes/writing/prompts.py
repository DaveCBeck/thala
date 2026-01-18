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

Target length: {_word_range(word_target)} words
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


def get_methodology_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate methodology system prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["methodology"])
    return f"""You are documenting the methodology for a systematic literature review.

Write a methodology section covering:
1. Search Strategy: Databases searched, query terms used
2. Selection Criteria: How papers were included/excluded
3. Data Extraction: What information was extracted from papers
4. Synthesis Approach: How findings were organized and synthesized

Target length: {_word_range(word_target)} words
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
2. Organize discussion by sub-themes or chronologically
3. Compare and contrast findings across papers
4. Note agreements, disagreements, and debates
5. Identify gaps and limitations
6. Use inline citations: [@CITATION_KEY] format

Target length: {_word_range(word_target)} words
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


def get_discussion_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate discussion system prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["discussion"])
    return f"""You are writing the discussion section for a systematic literature review.

The discussion should:
1. Synthesize findings ACROSS all themes (not repeat them)
2. Identify overarching patterns and trends
3. Discuss implications for theory and practice
4. Acknowledge limitations of this review
5. Suggest future research directions

Target length: {_word_range(word_target)} words
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
Avoid: Introducing new information or hedging excessively"""


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
