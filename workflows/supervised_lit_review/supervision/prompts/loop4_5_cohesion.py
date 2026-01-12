"""Prompts for Loop 4.5: Cohesion check after section editing."""

# =============================================================================
# Loop 4.5: Cohesion Check Prompt
# =============================================================================

LOOP4_5_COHESION_PROMPT = """You are reviewing a literature review after deep section editing to determine if it needs structural reorganization.

## Document
{document}

## Question
After the section-level edits, does this document need to return to structural reorganization (Loop 3), or is the structure sound?

Consider:
- Has the parallel editing introduced structural issues?
- Are sections now in illogical order?
- Is there new redundancy that needs consolidation?

## Important: Conservative Assessment

Only flag needs_restructuring=true if there are MAJOR structural issues:
- Sections in completely illogical order (e.g., conclusion before methodology)
- Significant content duplication across sections (>50% overlap)
- Missing critical sections entirely

DO NOT flag for minor issues that can be addressed in Loop 5:
- Small redundancies in phrasing
- Slightly awkward transitions
- Citation formatting issues

Return needs_restructuring=false for all minor issues. Loop 5 can address these.

Respond with needs_restructuring (true/false) and your reasoning."""
