"""Reference checking node for Loop 5."""

import logging
from typing import Any

from workflows.shared.llm_utils import get_structured_output
from ...types import Edit, DocumentEdits
from ...prompts import LOOP5_REF_CHECK_SYSTEM, LOOP5_REF_CHECK_USER
from ...utils import format_paper_summaries_with_budget, create_manifest_note, extract_citation_keys_from_text
from ...store_query import SupervisionStoreQuery
from ...tools import create_paper_tools
from langchain_tools.perplexity import check_fact

from .utils import calculate_dynamic_char_budget, estimate_loop5_request_tokens, format_citation_keys, HAIKU_MAX_TOKENS
from .fact_checking import select_model_tier_for_context

logger = logging.getLogger(__name__)


async def reference_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """Sequential reference checking across all sections with dynamic token budgeting."""
    sections = state["sections"]
    num_sections = len(sections)
    logger.info(f"Loop 5: Starting reference checking across {num_sections} sections")

    store_query = SupervisionStoreQuery(state["paper_summaries"])
    paper_tools = create_paper_tools(state["paper_summaries"], store_query)
    all_tools = paper_tools + [check_fact]
    zotero_keys = state.get("zotero_keys", {})

    all_edits: list[Edit] = state.get("all_edits", []).copy()
    all_todos: list[str] = []

    for section in sections:
        section_content = section["section_content"]

        cited_keys = extract_citation_keys_from_text(section_content)
        key_to_doi = {v: k for k, v in zotero_keys.items()}
        cited_dois = {key_to_doi.get(k) for k in cited_keys if k in key_to_doi}

        cited_zotero_keys = {
            doi: key for doi, key in zotero_keys.items()
            if key in cited_keys
        }
        citation_keys_text = format_citation_keys(cited_zotero_keys)

        cited_summaries = {
            doi: state["paper_summaries"][doi]
            for doi in cited_dois if doi in state["paper_summaries"]
        }

        dynamic_max_chars = calculate_dynamic_char_budget(
            section_content=section_content,
            system_prompt=LOOP5_REF_CHECK_SYSTEM,
            num_sections=num_sections,
            target_max_tokens=HAIKU_MAX_TOKENS,
        )

        detailed_content = await store_query.get_papers_for_section(
            section_content,
            compression_level=2,
            max_total_chars=dynamic_max_chars,
        )

        paper_summaries_text = format_paper_summaries_with_budget(
            cited_summaries,
            detailed_content,
            max_total_chars=dynamic_max_chars,
        )

        manifest_note = create_manifest_note(
            papers_with_detail=len(detailed_content),
            papers_total=len(cited_summaries),
            compression_level=2,
        )

        user_prompt = LOOP5_REF_CHECK_USER.format(
            section_content=section_content,
            citation_keys=citation_keys_text,
            paper_summaries=f"{manifest_note}\n\n{paper_summaries_text}",
        )

        estimated_tokens = estimate_loop5_request_tokens(
            section_content=section_content,
            system_prompt=LOOP5_REF_CHECK_SYSTEM,
            paper_summaries_text=f"{manifest_note}\n\n{paper_summaries_text}",
        )
        model_tier = select_model_tier_for_context(estimated_tokens)

        result = await get_structured_output(
            output_schema=DocumentEdits,
            user_prompt=user_prompt,
            system_prompt=LOOP5_REF_CHECK_SYSTEM,
            tools=all_tools,
            tier=model_tier,
            max_tokens=4096,
            max_tool_calls=12,
            max_tool_result_chars=100000,
        )

        all_edits.extend(result.edits)
        all_todos.extend(result.unaddressed_todos)

    logger.info(f"Loop 5: Reference check complete, total edits: {len(all_edits)}, unaddressed TODOs: {len(all_todos)}")
    return {
        "all_edits": all_edits,
        "unaddressed_todos": all_todos,
    }
