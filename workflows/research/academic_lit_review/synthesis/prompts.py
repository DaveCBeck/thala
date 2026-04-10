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
    SHARED_PROSE_CONSTRAINTS,
    SHARED_QUANTITATIVE_REMIT,
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

{SHARED_PROSE_CONSTRAINTS}

{SHARED_QUANTITATIVE_REMIT}

Output ONLY the abstract text, no headers or additional formatting."""


ABSTRACT_USER_TEMPLATE = """Write an abstract for this literature review.

Topic: {topic}

You are given the introduction and the full thematic sections. You are NOT given the discussion or the conclusions, because the abstract must not preview cross-theme findings that the Discussion is responsible for earning. Read everything before writing.

Introduction (for context):
{introduction}

<thematic_sections>
{thematic_content}
</thematic_sections>

Write a concise abstract that:
- States the scope and the specific question the review answers
- Names the key themes covered (briefly)
- Highlights the headline findings established within individual thematic sections (specific, not generic)
- Notes one or two implications that follow directly from the thematic evidence

What the abstract MUST NOT do:
- Do NOT state cross-theme findings that require reading two or more thematic sections together. Those belong to the Discussion. Example: if ecological benefits appear in one thematic section and institutional resistance appears in another, do NOT state their spatial or temporal coincidence here. The Discussion earns that.
- Do NOT announce an integration argument, a structural alignment, a geographic inversion, or a temporal inversion between themes. If a finding would only be visible to a reader who had read multiple thematic sections, it is not an abstract finding.
- Do NOT invent findings; everything in the abstract must be supported by the introduction or the thematic sections above.

The abstract should give the reader an honest preview of what each theme establishes on its own, leaving the cross-theme synthesis to be discovered in the Discussion."""
