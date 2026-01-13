"""Section-level rewriting for structural issues.

This module replaces the structured edit approach (Phase B) with direct
section rewriting. Instead of generating StructuralEdit specifications,
we simply rewrite the affected sections to fix each issue.

Key advantages:
- LLMs are better at rewriting than specifying edit operations
- No validation failures from missing replacement_text
- More natural handling of complex structural issues
"""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm, get_structured_output
from workflows.supervised_lit_review.supervision.types import (
    StructuralIssue,
    SectionRewriteResult,
    Loop3RewriteManifest,
)
from workflows.supervised_lit_review.supervision.prompts import (
    SECTION_REWRITE_SYSTEM,
    SECTION_REWRITE_USER,
    SECTION_REWRITE_SUMMARY_SYSTEM,
    SECTION_REWRITE_SUMMARY_USER,
)
from workflows.supervised_lit_review.supervision.utils import number_paragraphs

logger = logging.getLogger(__name__)

# Number of context paragraphs to include before/after the section
CONTEXT_PARAGRAPHS = 3


def extract_section_with_context(
    paragraph_mapping: dict[int, str],
    affected_paragraphs: list[int],
    context_size: int = CONTEXT_PARAGRAPHS,
) -> tuple[str, str, str, int, int]:
    """Extract section content with surrounding context.

    Args:
        paragraph_mapping: {paragraph_num: text} mapping
        affected_paragraphs: List of paragraph numbers to rewrite
        context_size: Number of paragraphs before/after to include as context

    Returns:
        Tuple of (context_before, section_content, context_after, start_para, end_para)
    """
    if not affected_paragraphs:
        return "", "", "", 0, 0

    # Get the range of paragraphs to extract
    start_para = min(affected_paragraphs)
    end_para = max(affected_paragraphs)

    # Get all paragraph numbers in order
    all_para_nums = sorted(paragraph_mapping.keys())
    if not all_para_nums:
        return "", "", "", start_para, end_para

    min_para = min(all_para_nums)
    max_para = max(all_para_nums)

    # Extract context before
    context_start = max(min_para, start_para - context_size)
    context_before_paras = []
    for p in range(context_start, start_para):
        if p in paragraph_mapping:
            context_before_paras.append(paragraph_mapping[p])
    context_before = "\n\n".join(context_before_paras)

    # Extract the section to rewrite
    section_paras = []
    for p in range(start_para, end_para + 1):
        if p in paragraph_mapping:
            section_paras.append(paragraph_mapping[p])
    section_content = "\n\n".join(section_paras)

    # Extract context after
    context_end = min(max_para, end_para + context_size)
    context_after_paras = []
    for p in range(end_para + 1, context_end + 1):
        if p in paragraph_mapping:
            context_after_paras.append(paragraph_mapping[p])
    context_after = "\n\n".join(context_after_paras)

    return context_before, section_content, context_after, start_para, end_para


async def rewrite_section_for_issue(
    issue: StructuralIssue | dict,
    paragraph_mapping: dict[int, str],
    topic: str,
    zotero_keys: dict[str, str] | None = None,
) -> tuple[SectionRewriteResult | None, str | None]:
    """Rewrite a section to fix a specific structural issue.

    Args:
        issue: The structural issue to fix
        paragraph_mapping: {paragraph_num: text} mapping
        topic: Research topic for context
        zotero_keys: DOI -> citation key mapping for citation validation

    Returns:
        Tuple of (SectionRewriteResult, None) on success, or
        (None, skip_reason) if rewrite failed/skipped
    """
    # Handle dict input (from state)
    if isinstance(issue, dict):
        issue_id = issue.get("issue_id", 0)
        issue_type = issue.get("issue_type", "unknown")
        severity = issue.get("severity", "moderate")
        description = issue.get("description", "")
        suggested_resolution = issue.get("suggested_resolution", "")
        affected_paragraphs = issue.get("affected_paragraphs", [])
    else:
        issue_id = issue.issue_id
        issue_type = issue.issue_type
        severity = issue.severity
        description = issue.description
        suggested_resolution = issue.suggested_resolution
        affected_paragraphs = issue.affected_paragraphs

    # Skip issues that are pure moves (can't fix by rewriting in place)
    if issue_type == "misplaced_content" and suggested_resolution == "move":
        logger.info(f"Skipping issue {issue_id}: pure move operation")
        return None, "pure_move_operation"

    # Extract section with context
    context_before, section_content, context_after, start_para, end_para = (
        extract_section_with_context(paragraph_mapping, affected_paragraphs)
    )

    if not section_content:
        logger.warning(f"Issue {issue_id}: No content found for paragraphs {affected_paragraphs}")
        return None, f"no_content_for_paragraphs_{affected_paragraphs}"

    # Build the prompt
    user_prompt = SECTION_REWRITE_USER.format(
        issue_id=issue_id,
        issue_type=issue_type,
        severity=severity,
        description=description,
        suggested_resolution=suggested_resolution,
        affected_paragraphs=", ".join(f"P{p}" for p in affected_paragraphs),
        context_before=context_before if context_before else "(Start of document)",
        section_content=section_content,
        context_after=context_after if context_after else "(End of document)",
    )

    try:
        # Use Sonnet for rewrites (faster, cheaper, good enough for this task)
        llm = get_llm(tier=ModelTier.SONNET)
        messages = [
            {"role": "system", "content": SECTION_REWRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        response = await llm.ainvoke(messages)
        rewritten_content = response.content.strip()

        # Basic validation: rewrite shouldn't be empty
        if not rewritten_content or len(rewritten_content) < 50:
            actual_len = len(rewritten_content) if rewritten_content else 0
            logger.warning(f"Issue {issue_id}: Rewrite too short ({actual_len} < 50 chars)")
            return None, f"rewrite_too_short_{actual_len}_chars"

        # Generate a change summary for audit
        changes_summary = await generate_change_summary(
            section_content, rewritten_content, description
        )

        logger.info(
            f"Issue {issue_id} ({issue_type}): Rewrote P{start_para}-P{end_para}, "
            f"original={len(section_content)} chars, new={len(rewritten_content)} chars"
        )

        return SectionRewriteResult(
            issue_id=issue_id,
            original_paragraphs=list(range(start_para, end_para + 1)),
            rewritten_content=rewritten_content,
            changes_summary=changes_summary,
            confidence=0.8,  # Could be made more dynamic based on issue type
        ), None

    except Exception as e:
        logger.error(f"Issue {issue_id}: Rewrite failed: {e}")
        return None, f"llm_error_{type(e).__name__}"


async def generate_change_summary(
    original_content: str,
    rewritten_content: str,
    issue_description: str,
) -> str:
    """Generate a brief summary of what changed for audit purposes."""
    try:
        # Use Haiku for this simple summarization task
        llm = get_llm(tier=ModelTier.HAIKU, max_tokens=500)
        user_prompt = SECTION_REWRITE_SUMMARY_USER.format(
            original_content=original_content[:2000],  # Truncate for token efficiency
            rewritten_content=rewritten_content[:2000],
            issue_description=issue_description,
        )
        messages = [
            {"role": "system", "content": SECTION_REWRITE_SUMMARY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Change summary generation failed: {e}")
        return f"Rewrite to fix: {issue_description[:100]}"


def apply_rewrite_to_document(
    paragraph_mapping: dict[int, str],
    rewrite: SectionRewriteResult,
) -> dict[int, str]:
    """Apply a section rewrite to the paragraph mapping.

    Args:
        paragraph_mapping: Current {paragraph_num: text} mapping
        rewrite: The rewrite result to apply

    Returns:
        New paragraph mapping with rewrite applied
    """
    if not rewrite.original_paragraphs:
        return paragraph_mapping

    start_para = min(rewrite.original_paragraphs)
    end_para = max(rewrite.original_paragraphs)

    # Build new mapping
    new_mapping = {}

    # Copy paragraphs before the rewritten section
    for p, text in paragraph_mapping.items():
        if p < start_para:
            new_mapping[p] = text

    # Insert the rewritten content as a single paragraph at start_para
    # Note: The rewritten content may have multiple logical paragraphs
    # but we treat it as replacing the range
    new_mapping[start_para] = rewrite.rewritten_content

    # Determine the shift for paragraphs after
    # Original: paragraphs start_para to end_para (inclusive) = end_para - start_para + 1 paragraphs
    # New: single paragraph at start_para
    # But the rewritten content may contain multiple paragraphs - we'll re-number later
    old_range_size = end_para - start_para + 1
    new_para_num = start_para + 1

    # Copy paragraphs after the rewritten section (shifted)
    for p in sorted(paragraph_mapping.keys()):
        if p > end_para:
            # Shift down by (old_range_size - 1) since we now have just 1 paragraph
            new_mapping[new_para_num] = paragraph_mapping[p]
            new_para_num += 1

    return new_mapping


def rebuild_document_from_mapping(paragraph_mapping: dict[int, str]) -> str:
    """Rebuild document text from paragraph mapping."""
    paragraphs = [
        paragraph_mapping[p]
        for p in sorted(paragraph_mapping.keys())
    ]
    return "\n\n".join(paragraphs)


async def rewrite_sections_for_issues_node(state: dict) -> dict[str, Any]:
    """Phase B replacement: Rewrite sections to fix identified issues.

    Processes issues serially, rewriting the affected section for each.
    Re-numbers paragraphs between rewrites to maintain accurate mapping.

    This replaces generate_edits_phase_b_node with a more natural approach
    that doesn't require structured edit specification.
    """
    numbered_doc = state["numbered_document"]
    paragraph_mapping = state["paragraph_mapping"]
    input_data = state["input"]
    issue_analysis = state.get("issue_analysis", {})
    topic = input_data.get("topic", "")

    issues = issue_analysis.get("issues", [])

    if not issues:
        logger.info("No issues to rewrite")
        return {
            "rewrite_manifest": Loop3RewriteManifest(
                rewrites=[],
                issues_addressed=[],
                issues_skipped=[],
                overall_assessment="No structural issues identified.",
            ).model_dump(),
            "current_review": state["current_review"],
        }

    # Process issues bottom-to-top (reverse order by max paragraph number)
    # This ensures paragraph number shifts from earlier rewrites don't
    # invalidate the paragraph references for later issues
    sorted_issues = sorted(
        issues,
        key=lambda x: max(x.get("affected_paragraphs", [0])),
        reverse=True,
    )

    rewrites: list[SectionRewriteResult] = []
    issues_addressed: list[int] = []
    issues_skipped: list[int] = []
    skip_reasons: dict[int, str] = {}

    # Working copy of paragraph mapping
    working_mapping = dict(paragraph_mapping)

    # Get zotero keys from state
    zotero_keys = state.get("zotero_keys", {})

    for issue in sorted_issues:
        issue_id = issue.get("issue_id", 0)

        # Rewrite section for this issue
        result, skip_reason = await rewrite_section_for_issue(
            issue=issue,
            paragraph_mapping=working_mapping,
            topic=topic,
            zotero_keys=zotero_keys,
        )

        if result:
            # Apply rewrite and update working mapping
            working_mapping = apply_rewrite_to_document(working_mapping, result)
            rewrites.append(result)
            issues_addressed.append(issue_id)
            logger.info(f"Applied rewrite for issue {issue_id}")
        else:
            issues_skipped.append(issue_id)
            if skip_reason:
                skip_reasons[issue_id] = skip_reason
            logger.info(f"Skipped issue {issue_id}: {skip_reason}")

    # Rebuild document from final mapping
    final_document = rebuild_document_from_mapping(working_mapping)

    # Build manifest
    manifest = Loop3RewriteManifest(
        rewrites=rewrites,
        issues_addressed=issues_addressed,
        issues_skipped=issues_skipped,
        skip_reasons=skip_reasons,
        overall_assessment=(
            f"Addressed {len(issues_addressed)} issues via section rewrites. "
            f"Skipped {len(issues_skipped)} issues (move operations or failures)."
        ),
    )

    logger.info(
        f"Section rewriting complete: {len(rewrites)} rewrites applied, "
        f"{len(issues_skipped)} skipped"
    )

    return {
        "rewrite_manifest": manifest.model_dump(),
        "current_review": final_document,
        "changes_applied": [r.changes_summary for r in rewrites],
    }
