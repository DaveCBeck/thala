"""Post-enhancement methodology transparency.

Appends an editorial process summary to the methodology section of the
final report, based on significant changes made during the editing phase.
Runs after enhancement completes — pure string manipulation, no LLM calls.
"""

import logging
import re

from workflows.enhance.editing.transparency import summarise_editorial_changes

logger = logging.getLogger(__name__)

# Pattern to find the methodology section heading in the review markdown
_METHODOLOGY_HEADING_RE = re.compile(
    r"^(#{1,3}\s+.*(?:Methodology|Methods|Research Design).*$)",
    re.MULTILINE | re.IGNORECASE,
)

# Pattern to find the next top-level heading after methodology
_NEXT_HEADING_RE = re.compile(r"^#{1,2}\s+", re.MULTILINE)


def append_editorial_summary(
    final_report: str,
    enhance_result: dict,
) -> str:
    """Append editorial transparency summary to the methodology section.

    If no significant editorial changes occurred, returns the report unchanged.

    Args:
        final_report: The final report markdown after enhancement.
        enhance_result: Result dict from enhance_report(), containing
            editing_state with section_enhancements.

    Returns:
        Report with editorial summary appended to methodology (if significant).
    """
    # Extract section enhancements from the editing result
    editing_state = enhance_result.get("editing_state", {})
    section_enhancements = editing_state.get("section_enhancements", [])

    if not section_enhancements:
        return final_report

    summary = summarise_editorial_changes(section_enhancements)
    if not summary:
        logger.debug("No significant editorial changes to report")
        return final_report

    editorial_section = f"\n\n### Editorial Process\n\n{summary}\n"

    # Find the methodology section and insert before the next top-level heading
    methodology_match = _METHODOLOGY_HEADING_RE.search(final_report)
    if not methodology_match:
        logger.warning("Could not find methodology section heading, appending editorial summary at end")
        return final_report + editorial_section

    # Find the next top-level heading after methodology
    search_start = methodology_match.end()
    next_heading = _NEXT_HEADING_RE.search(final_report[search_start:])

    if next_heading:
        insert_pos = search_start + next_heading.start()
        return final_report[:insert_pos] + editorial_section + "\n" + final_report[insert_pos:]
    else:
        # Methodology is the last section — append at end
        return final_report + editorial_section
