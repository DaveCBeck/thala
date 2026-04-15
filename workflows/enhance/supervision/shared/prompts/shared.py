"""Shared prompts used across multiple supervision loops."""


def build_word_budget_guidance(
    *,
    current_words: int,
    target_words: int,
    allowance: int,
    phase_label: str,
) -> str:
    """Render the standard word-budget block shown to integrator LLMs.

    Args:
        current_words: Word count of the input review/report.
        target_words: Target word count after this integration step.
        allowance: Permitted expansion from current_words (target - current).
        phase_label: Short label for this integration step, e.g.
            "Loop 1 theoretical-depth integration".
    """
    return f"""<Word Budget>
Phase: {phase_label}
Current review: ~{current_words:,} words
Target after this step: ~{target_words:,} words (allowance: +{allowance:,} words)

Aim for the target. Prefer integrating new material by weaving it into
existing sections over adding entire new sections. You MAY condense or
trim existing content to stay within budget, BUT ONLY when BOTH of the
following hold:
  1. The budget cannot be met by tighter prose in the new material alone.
  2. The new content answers the research questions better than the
     original detail it would displace.

Default bias: preserve existing analytical depth, citations, and
structure. If in doubt, keep the original content and tighten the new
material instead.
</Word Budget>"""


# =============================================================================
# Revision History Prompt
# =============================================================================

REVISION_SUMMARY_PROMPT = """Summarize the changes made to a document in one loop iteration.

## Loop Information
Loop {loop_number}, Iteration {iteration}

## Changes Summary
Lines added: {lines_added}
Lines removed: {lines_removed}
Sections modified: {sections_modified}

## Diff Sample (first 2000 chars)
{diff_sample}

Provide:
1. A brief summary (1-2 sentences) of what changed
2. A list of specific changes made (3-5 bullet points)
3. The reasoning behind these changes"""
