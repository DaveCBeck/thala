"""Reference checking node for Loop 5."""

import logging
from typing import Any

from workflows.shared.llm_utils import get_structured_output
from workflows.shared.token_utils import estimate_tokens_fast
from ...types import Edit, DocumentEdits
from ...prompts import LOOP5_REF_CHECK_SYSTEM, LOOP5_REF_CHECK_USER
from ...utils import extract_citation_keys_from_text
from ...store_query import SupervisionStoreQuery
from ...tools import create_paper_tools
from langchain_tools.perplexity import check_fact

from .fact_checking import select_model_tier_for_context

logger = logging.getLogger(__name__)


async def reference_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """Sequential reference checking across all sections."""
    sections = state["sections"]
    num_sections = len(sections)
    logger.info(f"Loop 5 reference checking: starting across {num_sections} sections")

    store_query = SupervisionStoreQuery()
    paper_tools = create_paper_tools(store_query)
    all_tools = paper_tools + [check_fact]

    all_edits: list[Edit] = state.get("all_edits", []).copy()
    all_todos: list[str] = []

    for section in sections:
        section_content = section["section_content"]

        # Extract citation keys from section for display
        cited_keys = extract_citation_keys_from_text(section_content)
        citation_keys_text = ", ".join(f"[@{k}]" for k in sorted(cited_keys)) if cited_keys else "None"

        user_prompt = LOOP5_REF_CHECK_USER.format(
            section_content=section_content,
            citation_keys=citation_keys_text,
        )

        # Estimate tokens for model selection
        estimated_tokens = estimate_tokens_fast(
            LOOP5_REF_CHECK_SYSTEM + user_prompt,
            with_safety_margin=True,
        )
        model_tier = select_model_tier_for_context(estimated_tokens)

        result = await get_structured_output(
            output_schema=DocumentEdits,
            user_prompt=user_prompt,
            system_prompt=LOOP5_REF_CHECK_SYSTEM,
            tools=all_tools,
            tier=model_tier,
            max_tokens=4096,
        )

        all_edits.extend(result.edits)
        all_todos.extend(result.unaddressed_todos)

    await store_query.close()

    logger.info(f"Loop 5 reference checking complete: {len(all_edits)} total edits, {len(all_todos)} unaddressed TODOs")
    return {
        "all_edits": all_edits,
        "unaddressed_todos": all_todos,
    }
