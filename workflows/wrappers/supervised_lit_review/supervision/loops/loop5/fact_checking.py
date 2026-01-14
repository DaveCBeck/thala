"""Fact checking node for Loop 5."""

import logging
from typing import Any

from workflows.shared.llm_utils import get_structured_output
from workflows.shared.llm_utils import ModelTier
from workflows.shared.token_utils import select_model_for_context, estimate_tokens_fast
from ...types import Edit, DocumentEdits
from ...prompts import LOOP5_FACT_CHECK_SYSTEM, LOOP5_FACT_CHECK_USER
from ...store_query import SupervisionStoreQuery
from ...tools import create_paper_tools
from langchain_tools.perplexity import check_fact

logger = logging.getLogger(__name__)


def select_model_tier_for_context(estimated_tokens: int) -> ModelTier:
    """Select appropriate model tier based on estimated context size."""
    model_name = select_model_for_context(estimated_tokens, prefer_haiku=True)

    if model_name == "sonnet_1m":
        logger.warning(
            f"Context size {estimated_tokens:,} tokens exceeds Haiku safe limit, using SONNET_1M"
        )
        return ModelTier.SONNET_1M
    elif model_name == "sonnet":
        logger.debug(f"Using SONNET for context size {estimated_tokens:,} tokens")
        return ModelTier.SONNET
    else:
        return ModelTier.HAIKU


async def fact_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """Sequential fact checking across all sections."""
    sections = state["sections"]
    num_sections = len(sections)
    logger.info(f"Loop 5 fact checking: starting across {num_sections} sections")

    store_query = SupervisionStoreQuery()
    paper_tools = create_paper_tools(store_query)
    all_tools = paper_tools + [check_fact]

    all_edits: list[Edit] = []
    all_ambiguous: list[str] = []

    for section in sections:
        section_content = section["section_content"]

        user_prompt = LOOP5_FACT_CHECK_USER.format(
            section_content=section_content,
        )

        # Estimate tokens for model selection
        estimated_tokens = estimate_tokens_fast(
            LOOP5_FACT_CHECK_SYSTEM + user_prompt,
            with_safety_margin=True,
        )
        model_tier = select_model_tier_for_context(estimated_tokens)

        result = await get_structured_output(
            output_schema=DocumentEdits,
            user_prompt=user_prompt,
            system_prompt=LOOP5_FACT_CHECK_SYSTEM,
            tools=all_tools,
            tier=model_tier,
            max_tokens=4096,
        )

        all_edits.extend(result.edits)
        all_ambiguous.extend(result.ambiguous_claims)

    await store_query.close()

    logger.info(
        f"Loop 5 fact checking complete: {len(all_edits)} edits, {len(all_ambiguous)} ambiguous claims"
    )
    return {
        "all_edits": all_edits,
        "ambiguous_claims": all_ambiguous,
    }
