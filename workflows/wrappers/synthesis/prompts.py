"""Prompt templates for synthesis workflow.

Section word targets are calculated as proportions of the total target word count:
- Introduction: 8%
- Body sections combined: 75% (divided by section count)
- Conclusion: 7%
- Simple synthesis uses the full target
"""

# Section proportions of total word count
SECTION_PROPORTIONS = {
    "introduction": 0.08,
    "body_total": 0.75,
    "conclusion": 0.07,
}

# Default total word count if not specified
DEFAULT_TARGET_WORDS = 18000


def _word_range(target: int, variance: float = 0.15) -> str:
    """Format a word count target as a range (e.g., '850-1050')."""
    low = int(target * (1 - variance))
    high = int(target * (1 + variance))
    return f"{low}-{high}"


def get_section_targets(total_words: int, section_count: int = 5) -> dict[str, int]:
    """Calculate word targets for each section based on total target.

    Args:
        total_words: Total target word count for the entire synthesis
        section_count: Number of body sections (affects per-section target)

    Returns:
        Dictionary mapping section names to their word count targets
    """
    section_count = max(1, section_count)
    body_per_section = int(total_words * SECTION_PROPORTIONS["body_total"] / section_count)

    return {
        "introduction": int(total_words * SECTION_PROPORTIONS["introduction"]),
        "body_section": body_per_section,
        "conclusion": int(total_words * SECTION_PROPORTIONS["conclusion"]),
    }


def get_section_writing_prompt(
    target_words: int = DEFAULT_TARGET_WORDS,
    section_count: int = 5,
) -> str:
    """Generate section writing prompt with appropriate word target.

    Args:
        target_words: Total target word count for the entire synthesis
        section_count: Number of body sections to divide the body budget across
    """
    section_count = max(1, section_count)
    word_target = int(target_words * SECTION_PROPORTIONS["body_total"] / section_count)
    return f"""Write a section for a synthesis document.

## Document Title
{{document_title}}

## Section: {{section_title}}
{{section_description}}

## Key Sources to Integrate
{{key_sources}}

## Available Research

### Academic Literature
{{lit_review_excerpt}}

### Web Research
{{web_research_excerpt}}

### Book Summaries
{{book_summaries_excerpt}}

## Writing Guidelines
1. Integrate insights from multiple sources
2. Use citations in [@ZOTKEY] format (e.g., [@SMITH2024])
3. Maintain academic tone while being accessible
4. Create clear transitions between ideas
5. Support claims with evidence from sources

Target length: {_word_range(word_target)} words

Write the complete section content in markdown format. Do NOT include the section title as a header - just write the content."""


def get_simple_synthesis_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate simple synthesis prompt with appropriate word target."""
    return f"""Create a comprehensive synthesis based on the following research.

## Topic
{{topic}}

## Research Questions
{{research_questions}}

## Academic Literature
{{lit_review_summary}}

## Web Research
{{web_research_summary}}

## Book Insights
{{book_summary}}

Write a comprehensive synthesis that:
1. Integrates all sources coherently
2. Addresses each research question thoroughly
3. Uses citations in [@ZOTKEY] format where available
4. Is well-structured with clear sections
5. Provides substantive analysis, not just summaries

Target length: {_word_range(target_words)} words

Output the complete synthesis document in markdown format."""


def get_introduction_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate introduction writing prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["introduction"])
    return f"""Write an introduction for the synthesis document.

## Document Title
{{document_title}}

## Topic
{{topic}}

## Research Questions
{{research_questions}}

## Guidance
{{intro_guidance}}

## Writing Guidelines
1. Introduce the topic and its significance
2. Preview the key themes and questions addressed
3. Set up the structure of the document
4. Engage the reader with the importance of the topic

Target length: {_word_range(word_target)} words

Write the introduction content in markdown format."""


def get_conclusion_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate conclusion writing prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["conclusion"])
    return f"""Write a conclusion for the synthesis document.

## Document Title
{{document_title}}

## Topic
{{topic}}

## Research Questions
{{research_questions}}

## Guidance
{{conclusion_guidance}}

## Key Themes Covered
{{themes_covered}}

## Writing Guidelines
1. Synthesize the main findings across all sections
2. Answer each research question definitively
3. Discuss implications and future directions
4. End with a compelling closing statement

Target length: {_word_range(word_target)} words

Write the conclusion content in markdown format."""


# Backwards compatibility: static prompts using default target
SECTION_WRITING_PROMPT = get_section_writing_prompt()
SIMPLE_SYNTHESIS_PROMPT = get_simple_synthesis_prompt()
