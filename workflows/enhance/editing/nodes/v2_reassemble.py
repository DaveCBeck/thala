"""V2 Phase 3: Reassembly and verification node.

This node reassembles the document from rewritten sections
and performs final coherence verification.
"""

import logging
from datetime import datetime
from typing import Any

from langsmith import traceable

from workflows.shared.llm_utils import ModelTier, get_structured_output

from ..prompts import V2_VERIFICATION_SYSTEM, V2_VERIFICATION_USER
from ..schemas import V2FinalVerification, RewrittenSection, TopLevelSection

logger = logging.getLogger(__name__)


def build_changes_summary(
    sections: list[TopLevelSection],
    rewritten: list[RewrittenSection],
) -> str:
    """Build a human-readable summary of changes made.

    Args:
        sections: Original sections
        rewritten: Rewritten sections

    Returns:
        Formatted summary string
    """
    if not rewritten:
        return "No changes made to the document."

    lines = ["## Changes Summary", ""]

    for rw in sorted(rewritten, key=lambda x: x.section_index):
        section = sections[rw.section_index]
        validation = rw.validation

        if rw.instruction_type == "delete":
            lines.append(f"- **Deleted** section [{rw.section_index}]: '{section.heading}'")
        elif rw.instruction_type == "merge_into" and rw.merge_source_index is not None:
            source = sections[rw.merge_source_index]
            lines.append(
                f"- **Merged** section [{rw.merge_source_index}] '{source.heading}' "
                f"into [{rw.section_index}] '{section.heading}'"
            )
            lines.append(
                f"  - Word count: {validation.original_word_count} -> "
                f"{validation.rewritten_word_count}"
            )
        else:
            lines.append(
                f"- **{rw.instruction_type.capitalize()}** section [{rw.section_index}]: "
                f"'{section.heading}'"
            )
            lines.append(
                f"  - Word count: {validation.original_word_count} -> "
                f"{validation.rewritten_word_count} "
                f"({validation.length_ratio:.1%})"
            )

        if not validation.passes_validation:
            lines.append(f"  - **Warning**: {validation.rejection_reason}")

    # Summary stats
    lines.append("")
    lines.append(f"Total sections modified: {len(rewritten)}")

    # Count by type
    by_type: dict[str, int] = {}
    for rw in rewritten:
        by_type[rw.instruction_type] = by_type.get(rw.instruction_type, 0) + 1
    for instr_type, count in sorted(by_type.items()):
        lines.append(f"  - {instr_type}: {count}")

    return "\n".join(lines)


@traceable(run_type="chain", name="EditingV2.Reassemble")
async def v2_reassemble_node(state: dict) -> dict[str, Any]:
    """V2 Phase 3: Reassemble document and verify coherence.

    This node:
    1. Assembles the final document from original + rewritten sections
    2. Handles deletions and merges
    3. Verifies overall coherence
    4. Produces change summary

    Args:
        state: Current workflow state

    Returns:
        State update with final_document, changes_summary, verification
    """
    # Extract from state
    sections_data = state.get("sections", [])
    rewritten_data = state.get("rewritten_sections", [])
    input_data = state["input"]
    topic = input_data["topic"]

    # Reconstruct objects
    sections = [TopLevelSection(**s) for s in sections_data]
    rewritten = [RewrittenSection(**r) for r in rewritten_data]

    if not sections:
        logger.error("No sections found in state")
        return {
            "final_document": state["input"]["document"],
            "changes_summary": "Error: No sections found",
            "verification": V2FinalVerification(
                coherence_score=0.0,
                flow_assessment="Error during reassembly",
                issues_found=["No sections found in document"],
                recommendation="reject",
            ).model_dump(),
            "status": "failed",
            "completed_at": datetime.utcnow(),
        }

    logger.info(f"Reassembling document: {len(sections)} sections, {len(rewritten)} rewrites")

    # Build index of rewrites by section index
    rewrites_by_index: dict[int, RewrittenSection] = {rw.section_index: rw for rw in rewritten}

    # Track sections that were merged into others (should be removed)
    sections_to_remove: set[int] = set()
    for rw in rewritten:
        if rw.instruction_type == "merge_into" and rw.merge_source_index is not None:
            sections_to_remove.add(rw.merge_source_index)
        elif rw.instruction_type == "delete":
            sections_to_remove.add(rw.section_index)

    # Assemble final document
    final_parts: list[str] = []

    for section in sections:
        idx = section.index

        # Skip sections that were merged/deleted
        if idx in sections_to_remove:
            logger.info(f"Skipping removed section [{idx}]: '{section.heading}'")
            continue

        # Use rewritten content if available, otherwise keep original
        if idx in rewrites_by_index:
            rw = rewrites_by_index[idx]
            if rw.instruction_type != "delete":  # Double-check
                final_parts.append(rw.new_content)
                logger.debug(f"Using rewritten content for section [{idx}]")
        else:
            final_parts.append(section.full_content)
            logger.debug(f"Keeping original content for section [{idx}]")

    final_document = "\n\n".join(final_parts)

    # Build changes summary
    changes_summary = build_changes_summary(sections, rewritten)

    # Verify coherence if there were changes
    if rewritten:
        try:
            verification = await get_structured_output(
                output_schema=V2FinalVerification,
                user_prompt=V2_VERIFICATION_USER.format(
                    topic=topic,
                    document=final_document,
                    changes_summary=changes_summary,
                ),
                system_prompt=V2_VERIFICATION_SYSTEM,
                tier=ModelTier.SONNET,  # Use Sonnet for verification (faster)
                max_tokens=2000,
                use_json_schema_method=True,
                max_retries=2,
            )
        except Exception as e:
            logger.error(f"Verification LLM call failed: {e}")
            verification = V2FinalVerification(
                coherence_score=0.5,
                flow_assessment=f"Verification failed: {e}",
                issues_found=["Could not complete coherence verification"],
                recommendation="review",
            )
    else:
        # No changes made
        verification = V2FinalVerification(
            coherence_score=1.0,
            flow_assessment="No changes were needed",
            issues_found=[],
            recommendation="accept",
        )

    logger.info(
        f"Reassembly complete: coherence={verification.coherence_score:.2f}, "
        f"recommendation={verification.recommendation}"
    )

    # Determine final status
    if verification.recommendation == "reject":
        status = "partial"
    elif any(not rw.validation.passes_validation for rw in rewritten):
        status = "partial"
    else:
        status = "success"

    return {
        "final_document": final_document,
        "changes_summary": changes_summary,
        "verification": verification.model_dump(),
        "rewriting_complete": True,
        "status": status,
        "completed_at": datetime.utcnow(),
    }
