"""Loop 4.5: Cohesion check after section editing."""

from typing import Any

from workflows.shared.llm_utils.models import ModelTier, get_llm
from workflows.research.subgraphs.academic_lit_review.supervision.types import (
    CohesionCheckResult,
)
from workflows.research.subgraphs.academic_lit_review.supervision.prompts import (
    LOOP4_5_COHESION_PROMPT,
)


async def check_cohesion(document: str) -> CohesionCheckResult:
    """Check if document needs structural reorganization after Loop 4 edits.

    Single Opus call with structured output to determine if parallel editing
    introduced structural issues requiring a return to Loop 3.

    Args:
        document: The document after Loop 4 section editing

    Returns:
        CohesionCheckResult with needs_restructuring bool and reasoning
    """
    llm = get_llm(
        tier=ModelTier.OPUS,
        thinking_budget=4000,
        max_tokens=4096,
    )

    # Get structured output
    structured_llm = llm.with_structured_output(CohesionCheckResult)

    prompt = LOOP4_5_COHESION_PROMPT.format(document=document)

    result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])

    return result


async def run_loop4_5_standalone(document: str) -> dict[str, Any]:
    """Run Loop 4.5 as standalone operation for testing.

    Args:
        document: Document to check for cohesion

    Returns:
        Dict with needs_restructuring, reasoning
    """
    result = await check_cohesion(document)
    return {
        "needs_restructuring": result.needs_restructuring,
        "reasoning": result.reasoning,
    }
