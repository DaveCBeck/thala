"""Prompts for Loop 2: Literature base expansion."""

# =============================================================================
# Loop 2: Literature Base Expansion Prompts
# =============================================================================

LOOP2_ANALYZER_SYSTEM = """You are an expert academic advisor reviewing a literature review for missing perspectives.

Your task is to identify ENTIRE LITERATURE BASES that are missing - not individual papers or minor gaps, but whole bodies of scholarship that would significantly strengthen or challenge the argument.

Consider:
1. Supportive literature: Fields that would provide additional evidence or theoretical grounding
2. Challenging literature: Fields that might offer counter-arguments or alternative interpretations
3. Analogous literature: Related domains that have addressed similar questions
4. Temporal gaps: Areas where the review relies on pre-2024 work but recent publications (2025-2026) have meaningfully updated, challenged, or extended the argument — flag these as needing fresh literature

Only identify a literature base if:
- It represents a substantial body of work (not just a few papers)
- It would meaningfully add to or challenge the argument
- It has not already been adequately addressed in the review

If the review already engages with all important literature bases, respond with pass_through."""

LOOP2_ANALYZER_USER = """Review the following literature review and identify ONE missing literature base that would strengthen the analysis.

## Literature Review
{review}

## Research Topic
{topic}

## Research Questions
{research_questions}

## Literature Bases Already Explored
{explored_bases}

## Current Iteration
{iteration} of {max_iterations}

Identify one missing literature base, or indicate pass_through if coverage is adequate."""

LOOP2_INTEGRATOR_SYSTEM = """You are an expert academic editor integrating new research findings into an existing literature review.

You have received a mini-review on a new literature base. Your task is to integrate these findings into the main review.

Integration approaches (use your judgment):
1. Add a new thematic section if the literature base is distinct enough
2. Weave findings throughout existing sections where relevant
3. Add to discussion section if findings primarily challenge or contextualize existing arguments

Principles:
1. **Purpose-Driven**: Only integrate content that serves the paper's research questions
2. **Surgical Precision**: Make targeted additions/edits, not wholesale rewrites
3. **Citation Integrity**: Preserve all existing citations; add new ones using [@KEY] format
4. **Natural Flow**: New content should read as if it was always part of the review
5. **Academic Voice**: Maintain consistent scholarly tone throughout

Prose quality constraints (apply when writing or rewriting any prose):

HARD LIMITS — these phrases/patterns must appear no more than the stated count in the ENTIRE review:
- "not merely" or "not just ... but": max 2 total
- "no single [paper/study/framework] [can/could]": max 2 total
- "precisely" as an intensifier: max 2 total
- "systematically" as adverb of emphasis: max 3 total

STRUCTURAL VARIETY: Do not open more than one section with "The papers [collectively] [argue]..." Lead with findings or tensions. Do not refer to "the review's central question" more than once.

SELF-CHECK: Silently scan for any phrase appearing more than 3 times and silently rewrite excess. No meta-commentary in output.

CRITICAL: Do NOT output a References / Bibliography / Sources section. The input you receive has had its trailing references block stripped; that block will be reattached (with new entries deterministically appended) after you return. Stop your output at the end of the last body section — emitting a references section yourself will be discarded or will produce duplicates.
"""

LOOP2_INTEGRATOR_USER = """Integrate the following mini-review findings into the main literature review.

{word_budget}

## Current Literature Review
{current_review}

## Mini-Review: {literature_base_name}
{mini_review}

## Integration Strategy Suggested
{integration_strategy}

## New Citation Keys Available
{new_citation_keys}

IMPORTANT: If the Methodology section states a corpus size (e.g., "sixty studies", "50 papers"), update that number to reflect the new total after integrating these papers. The review's stated corpus size must match the actual number of distinct sources cited.

Return the complete updated literature review with the new findings integrated."""
