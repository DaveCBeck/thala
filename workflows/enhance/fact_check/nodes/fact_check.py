"""Fact-check section worker for fact-check workflow."""

import logging
from typing import Any

from langsmith import traceable
from langgraph.types import Send

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.fact_check.schemas import (
    FactCheckResult,
)
from workflows.enhance.fact_check.prompts import (
    FACT_CHECK_SYSTEM,
    FACT_CHECK_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


def route_to_fact_check_sections(state: dict) -> list[Send] | str:
    """Route to fact-check workers for screened sections only.

    Uses pre-screening results to only check sections with verifiable claims.
    Falls back to checking all sections if screening wasn't performed.

    Returns list of Send objects for parallel fact-checking,
    or "reference_check_router" if no sections to check.
    """
    document_model_dict = state.get("updated_document_model", state.get("document_model"))
    if not document_model_dict:
        return "reference_check_router"

    document_model = DocumentModel.from_dict(document_model_dict)
    quality_settings = state.get("quality_settings", {})
    use_perplexity = quality_settings.get("verify_use_perplexity", True)
    max_tool_calls = quality_settings.get("fact_check_max_tool_calls", 15)

    # Get sections to check from screening results
    screened_sections = state.get("screened_sections", [])
    screening_skipped = state.get("screening_skipped", [])

    # Get all leaf sections with content
    all_sections = document_model.get_all_sections()
    leaf_sections = [s for s in all_sections if not s.subsections and s.blocks]

    # If screening was performed, use its results
    if screened_sections or screening_skipped:
        sections_to_check = [
            s for s in leaf_sections
            if s.section_id in screened_sections
        ]
        logger.info(
            f"Using screening results: {len(sections_to_check)} sections to check, "
            f"{len(screening_skipped)} skipped"
        )
    else:
        # Fallback: check all sections (screening was skipped or failed)
        sections_to_check = leaf_sections
        logger.info(f"No screening results, checking all {len(sections_to_check)} sections")

    if not sections_to_check:
        return "reference_check_router"

    # Build Send objects for parallel fact-checking
    sends = []
    for section in sections_to_check:
        sends.append(
            Send(
                "fact_check_section",
                {
                    "section_id": section.section_id,
                    "section_content": document_model.get_section_content(
                        section.section_id, include_subsections=False
                    ),
                    "section_heading": section.heading,
                    "topic": state["input"]["topic"],
                    "use_perplexity": use_perplexity,
                    "confidence_threshold": quality_settings.get("verify_confidence_threshold", 0.75),
                    "max_tool_calls": max_tool_calls,
                },
            )
        )

    logger.info(f"Routing to fact-check {len(sends)} sections (perplexity={use_perplexity})")
    return sends


@traceable(run_type="chain", name="FactCheckSectionWorker")
async def fact_check_section_worker(state: dict) -> dict[str, Any]:
    """Fact-check claims in a single section.

    This worker:
    1. Identifies factual claims in the section
    2. Verifies each claim using corpus search (and Perplexity if enabled)
    3. Suggests edits for claims that need correction
    4. Returns verification results with suggested edits
    """
    section_id = state["section_id"]
    section_content = state["section_content"]
    section_heading = state["section_heading"]
    topic = state["topic"]
    use_perplexity = state.get("use_perplexity", True)
    confidence_threshold = state.get("confidence_threshold", 0.75)
    max_tool_calls = state.get("max_tool_calls", 15)

    logger.debug(
        f"Fact-checking section '{section_heading}' (perplexity={use_perplexity}, max_tools={max_tool_calls})"
    )

    # Get tools based on settings
    from langchain_tools import search_papers, get_paper_content
    tools = [search_papers, get_paper_content]

    if use_perplexity:
        from langchain_tools import check_fact
        tools.append(check_fact)

    user_prompt = FACT_CHECK_USER.format(
        section_heading=section_heading,
        section_content=section_content,
        topic=topic,
        use_perplexity="available" if use_perplexity else "not available",
        confidence_threshold=confidence_threshold,
    )

    try:
        result = await get_structured_output(
            output_schema=FactCheckResult,
            user_prompt=user_prompt,
            system_prompt=FACT_CHECK_SYSTEM,
            tier=ModelTier.HAIKU,
            tools=tools,
            max_tokens=4000,
            max_tool_calls=max_tool_calls,
            use_json_schema_method=True,
        )

        # Override section_id from result to match input
        result.section_id = section_id

        # Log unresolved issues at INFO level
        if result.unresolved_issues:
            for issue in result.unresolved_issues:
                logger.info(f"Unresolved fact-check issue in '{section_heading}': {issue}")

        # Filter suggested edits by confidence threshold
        valid_edits = [
            e for e in result.suggested_edits
            if e.confidence >= confidence_threshold
        ]
        low_confidence_edits = [
            e for e in result.suggested_edits
            if e.confidence < confidence_threshold
        ]

        if low_confidence_edits:
            logger.info(
                f"Skipping {len(low_confidence_edits)} low-confidence edits in '{section_heading}'"
            )
            # Add to unresolved issues
            for edit in low_confidence_edits:
                result.unresolved_issues.append(
                    f"Low-confidence edit skipped: {edit.justification} (confidence={edit.confidence:.2f})"
                )

        result.suggested_edits = valid_edits

        logger.info(
            f"Fact-checked section '{section_heading}': "
            f"{result.claims_checked} claims, {len(valid_edits)} edits suggested"
        )

        return {
            "fact_check_results": [result.model_dump()],
            "pending_edits": [e.model_dump() for e in valid_edits],
        }

    except Exception as e:
        logger.error(f"Fact-check failed for section '{section_heading}': {e}")
        return {
            "fact_check_results": [
                FactCheckResult(
                    section_id=section_id,
                    claims_checked=0,
                    claims_verified=[],
                    suggested_edits=[],
                    unresolved_issues=[f"Fact-check failed: {e}"],
                ).model_dump()
            ],
            "errors": [{"node": "fact_check_section", "error": str(e)}],
        }


@traceable(run_type="chain", name="FactCheckAssemble")
async def assemble_fact_checks_node(state: dict) -> dict[str, Any]:
    """Assemble fact-check results from parallel workers.

    Collects all pending edits and unresolved issues.
    """
    fact_check_results = state.get("fact_check_results", [])

    total_claims = sum(r.get("claims_checked", 0) for r in fact_check_results)
    total_edits = sum(len(r.get("suggested_edits", [])) for r in fact_check_results)
    total_unresolved = sum(len(r.get("unresolved_issues", [])) for r in fact_check_results)

    logger.info(
        f"Assembled fact-checks: {total_claims} claims checked, "
        f"{total_edits} edits pending, {total_unresolved} unresolved issues"
    )

    # Collect all unresolved items
    unresolved_items = []
    for result in fact_check_results:
        for issue in result.get("unresolved_issues", []):
            unresolved_items.append({
                "source": "fact_check",
                "section_id": result.get("section_id"),
                "issue": issue,
            })

    return {
        "unresolved_items": unresolved_items,
    }
