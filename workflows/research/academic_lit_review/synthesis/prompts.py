"""Prompt templates for synthesis subgraph.

Section word targets are calculated as proportions of the total target word count:
- Introduction: 8%
- Methodology: 6%
- Thematic sections combined: 70% (divided by theme count)
- Discussion: 9%
- Conclusions: 5%
- Abstract: 2% (handled in integration)
"""

# Re-export from writing prompts for backwards compatibility
from .nodes.writing.prompts import (  # noqa: F401
    SECTION_PROPORTIONS,
    DEFAULT_TARGET_WORDS,
    get_section_targets,
    get_introduction_system_prompt,
    get_methodology_system_prompt,
    get_thematic_section_system_prompt,
    get_discussion_system_prompt,
    get_conclusions_system_prompt,
    INTRODUCTION_SYSTEM_PROMPT,
    INTRODUCTION_USER_TEMPLATE,
    METHODOLOGY_SYSTEM_PROMPT,
    METHODOLOGY_USER_TEMPLATE,
    THEMATIC_SECTION_SYSTEM_PROMPT,
    THEMATIC_SECTION_USER_TEMPLATE,
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_USER_TEMPLATE,
    CONCLUSIONS_SYSTEM_PROMPT,
    CONCLUSIONS_USER_TEMPLATE,
    _word_range,
)


QUALITY_CHECK_SYSTEM_PROMPT = """You are reviewing a literature review for quality issues.

Check for:
1. Missing or unclear citations (claims without [@KEY])
2. Logical inconsistencies
3. Unsupported claims
4. Poor transitions
5. Redundancy
6. Balance across sections

Provide your assessment of the review quality."""


# Abstract generation prompts (used by programmatic integration)
def get_abstract_system_prompt(abstract_target: int = 240) -> str:
    """Generate system prompt for abstract-only generation."""
    return f"""You are writing an abstract for an academic literature review.

Write a concise abstract of {_word_range(abstract_target)} words that:
1. States the research topic and scope
2. Summarizes the methodology (literature search approach)
3. Highlights the key themes and findings
4. Notes the main conclusions and implications

The abstract should be self-contained and give readers a clear overview
of what the literature review covers and its main contributions.

Output ONLY the abstract text, no headers or additional formatting."""


ABSTRACT_USER_TEMPLATE = """Write an abstract for this literature review.

Topic: {topic}

You are given the introduction, the full thematic sections, the discussion, and the conclusions. The abstract must accurately preview the review's actual findings — not generic descriptions of the topic. Read everything before writing.

Introduction (for context):
{introduction}

<thematic_sections>
{thematic_content}
</thematic_sections>

<discussion>
{discussion}
</discussion>

<conclusions>
{conclusions}
</conclusions>

Write a concise abstract that:
- States the scope and the specific question the review answers
- Names the key themes covered (briefly)
- Highlights the most important findings the synthesis established (specific, not generic)
- Notes the main implications

The abstract should give a reader a clear, accurate preview of what the review actually establishes — not just what topic it covers. Do not invent findings; everything in the abstract must be supported by the sections above."""
