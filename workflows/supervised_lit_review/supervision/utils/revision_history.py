"""Revision history tracking for supervision loops."""

import difflib
import re

from workflows.shared.llm_utils import get_llm, ModelTier
from workflows.academic_lit_review.state import RevisionRecord


async def document_revision(
    loop_number: int,
    iteration: int,
    before_text: str,
    after_text: str,
) -> RevisionRecord:
    """Document changes made during a supervision loop iteration.

    Computes diff statistics and generates an LLM summary of changes.

    Args:
        loop_number: Which supervision loop (1-5)
        iteration: Iteration number within the loop
        before_text: Document text before changes
        after_text: Document text after changes

    Returns:
        RevisionRecord with diff stats and summary
    """
    # Compute basic diff statistics
    before_lines = before_text.split("\n")
    after_lines = after_text.split("\n")

    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=""))

    lines_added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    lines_removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    # Identify changed sections (look for markdown headers)
    changed_sections = _identify_changed_sections(before_text, after_text)

    # Generate summary using Haiku (async)
    summary = await _generate_change_summary(
        before_text, after_text, loop_number, lines_added, lines_removed, changed_sections
    )

    return RevisionRecord(
        loop_number=loop_number,
        iteration=iteration,
        summary=summary,
        changes_made=changed_sections,
        reasoning=f"Loop {loop_number} iteration {iteration}: {lines_added} lines added, {lines_removed} lines removed",
    )


def _identify_changed_sections(before_text: str, after_text: str) -> list[str]:
    """Identify which markdown sections have changed."""
    before_sections = _extract_section_headings(before_text)
    after_sections = _extract_section_headings(after_text)

    # Find sections that appear in one but not the other, or have different content
    changed = []

    # Check for new sections
    for section in after_sections:
        if section not in before_sections:
            changed.append(f"Added: {section}")

    # Check for removed sections
    for section in before_sections:
        if section not in after_sections:
            changed.append(f"Removed: {section}")

    # If no structural changes, just note that content was modified
    if not changed:
        changed.append("Content modifications within existing sections")

    return changed


def _extract_section_headings(text: str) -> list[str]:
    """Extract all markdown headings from text."""
    headings = []
    for line in text.split("\n"):
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            headings.append(match.group(2).strip())
    return headings


async def _generate_change_summary(
    before_text: str,
    after_text: str,
    loop_number: int,
    lines_added: int,
    lines_removed: int,
    changed_sections: list[str],
) -> str:
    """Generate LLM summary of changes (async)."""
    llm = get_llm(ModelTier.HAIKU)

    # Truncate texts if too long for context
    max_chars = 8000
    before_truncated = before_text[:max_chars] + ("..." if len(before_text) > max_chars else "")
    after_truncated = after_text[:max_chars] + ("..." if len(after_text) > max_chars else "")

    prompt = f"""Summarize the changes made to this academic literature review in Loop {loop_number}.

Statistics:
- Lines added: {lines_added}
- Lines removed: {lines_removed}
- Changed sections: {', '.join(changed_sections)}

BEFORE:
{before_truncated}

AFTER:
{after_truncated}

Provide a 2-3 sentence summary focusing on:
1. What type of changes were made (structural, content, citations, etc.)
2. The academic purpose/improvement from these changes

Keep the summary concise and focused on the substantive changes."""

    response = await llm.ainvoke(prompt)
    return response.content.strip()
