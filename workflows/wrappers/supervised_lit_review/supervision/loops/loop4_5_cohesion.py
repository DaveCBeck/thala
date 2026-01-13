"""Loop 4.5: Cohesion check after section editing."""

from typing import Any

from langsmith import traceable

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.wrappers.supervised_lit_review.supervision.types import (
    CohesionCheckResult,
)
from workflows.wrappers.supervised_lit_review.supervision.prompts import (
    LOOP4_5_COHESION_PROMPT,
)


@traceable(run_type="chain", name="Loop4_5_CohesionCheck")
async def check_cohesion(document: str) -> CohesionCheckResult:
    """Check if document needs structural reorganization after Loop 4 edits.

    Single Opus call with structured output to determine if parallel editing
    introduced structural issues requiring a return to Loop 3.

    Args:
        document: The document after Loop 4 section editing

    Returns:
        CohesionCheckResult with needs_restructuring bool and reasoning
    """
    import logging
    logger = logging.getLogger(__name__)

    prompt = LOOP4_5_COHESION_PROMPT.format(document=document)

    try:
        result = await get_structured_output(
            output_schema=CohesionCheckResult,
            user_prompt=prompt,
            tier=ModelTier.OPUS,
            thinking_budget=4000,
            max_tokens=4096,
            use_json_schema_method=True,
            max_retries=2,
        )
        return result

    except Exception as e:
        logger.error(f"Cohesion check failed: {e}")
        return CohesionCheckResult(
            needs_restructuring=False,
            reasoning=f"Analysis failed: {e}",
        )


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
