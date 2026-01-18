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


def get_integration_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate integration system prompt with appropriate abstract word target."""
    abstract_target = int(target_words * SECTION_PROPORTIONS["abstract"])
    return f"""You are integrating sections into a cohesive literature review document.

Your task:
1. Add smooth transitions between sections
2. Ensure consistent terminology throughout
3. Add an abstract ({_word_range(abstract_target)} words)
4. Format with proper markdown headers
5. Ensure logical flow

CRITICAL - CITATION PRESERVATION:
- You MUST preserve ALL citations in their exact [@KEY] format
- Do NOT change [@ABC123] to any other format
- Do NOT remove or omit any citations
- Do NOT convert to numbered references [1], [2], etc.
- Do NOT convert to inline (Author, Year) format
- The [@KEY] citations will be processed into a reference list automatically

Do NOT change the substantive content of any section.
Focus ONLY on transitions, section ordering, and adding the abstract.

Output the complete document with this structure:
# [Title]

## Abstract
[{_word_range(abstract_target)} word abstract]

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

Provide your assessment of the review quality."""


# Backwards compatibility: static prompt using default target
INTEGRATION_SYSTEM_PROMPT = get_integration_system_prompt()
