"""Shared prompts used across multiple supervision loops."""

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
