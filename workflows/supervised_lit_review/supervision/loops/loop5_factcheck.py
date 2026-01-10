"""Loop 5: Fact and reference checking with tool access."""

import logging
from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from core.stores.zotero import ZoteroStore
from workflows.shared.llm_utils import ModelTier, get_structured_output
from .todo_verification import verify_todos

logger = logging.getLogger(__name__)

# Token budgeting constants
CHARS_PER_TOKEN = 4  # Conservative estimate
HAIKU_MAX_TOKENS = 200_000
SONNET_1M_MAX_TOKENS = 800_000  # Safe limit for 1M context
SONNET_1M_THRESHOLD = 150_000  # Switch to Sonnet 1M if estimated tokens exceed this
SYSTEM_PROMPT_TOKENS = 1000  # Buffer for system prompt
RESPONSE_BUFFER_TOKENS = 4096  # Buffer for model response


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def estimate_loop5_request_tokens(
    section_content: str,
    system_prompt: str,
    paper_summaries_text: str,
) -> int:
    """Estimate total tokens for a Loop 5 LLM request."""
    section_tokens = estimate_tokens(section_content)
    system_tokens = estimate_tokens(system_prompt)
    paper_tokens = estimate_tokens(paper_summaries_text)
    return section_tokens + system_tokens + paper_tokens + RESPONSE_BUFFER_TOKENS


def calculate_dynamic_char_budget(
    section_content: str,
    system_prompt: str,
    num_sections: int,
    target_max_tokens: int = HAIKU_MAX_TOKENS,
) -> int:
    """Calculate dynamic character budget for paper context per section."""
    section_tokens = estimate_tokens(section_content)
    system_tokens = estimate_tokens(system_prompt)
    base_tokens = section_tokens + system_tokens + RESPONSE_BUFFER_TOKENS

    available_tokens = target_max_tokens - base_tokens
    per_section_tokens = int(available_tokens * 0.8 / max(num_sections, 1))
    available_chars = per_section_tokens * CHARS_PER_TOKEN

    return max(5000, min(available_chars, 20000))


def select_model_tier_for_context(estimated_tokens: int) -> ModelTier:
    """Select appropriate model tier based on estimated context size."""
    if estimated_tokens > SONNET_1M_THRESHOLD:
        logger.warning(
            f"Context size {estimated_tokens:,} tokens exceeds threshold "
            f"({SONNET_1M_THRESHOLD:,}), using SONNET_1M"
        )
        return ModelTier.SONNET_1M
    return ModelTier.HAIKU


from ..types import Edit, DocumentEdits
from ..prompts import (
    LOOP5_FACT_CHECK_SYSTEM,
    LOOP5_FACT_CHECK_USER,
    LOOP5_REF_CHECK_SYSTEM,
    LOOP5_REF_CHECK_USER,
)
from ..utils import (
    split_into_sections,
    validate_edits,
    apply_edits,
    SectionInfo,
    EditValidationResult,
    format_paper_summaries_with_budget,
    create_manifest_note,
    extract_citation_keys_from_text,
    validate_citations_against_zotero,
    strip_invalid_citations,
)
from ..store_query import SupervisionStoreQuery
from ..tools import create_paper_tools


class Loop5State(TypedDict):
    """State for Loop 5 fact and reference checking."""

    current_review: str
    paper_summaries: dict[str, Any]
    zotero_keys: dict[str, str]
    sections: list[SectionInfo]
    all_edits: list[Edit]
    valid_edits: list[Edit]
    invalid_edits: list[Edit]
    ambiguous_claims: list[str]
    unaddressed_todos: list[str]
    human_review_items: list[str]
    discarded_todos: list[str]
    iteration: int
    max_iterations: int
    is_complete: bool
    topic: str
    verify_todos_enabled: bool
    verify_zotero: bool
    verified_citation_keys: set[str]


def split_sections_node(state: Loop5State) -> dict[str, Any]:
    """Split document into sections for sequential checking."""
    sections = split_into_sections(state["current_review"])
    logger.info(f"Loop 5: Split document into {len(sections)} sections for checking")
    return {"sections": sections}


async def fact_check_node(state: Loop5State) -> dict[str, Any]:
    """Sequential fact checking across all sections with dynamic token budgeting."""
    sections = state["sections"]
    num_sections = len(sections)
    logger.info(f"Loop 5: Starting fact checking across {num_sections} sections")

    store_query = SupervisionStoreQuery(state["paper_summaries"])
    paper_tools = create_paper_tools(state["paper_summaries"], store_query)
    zotero_keys = state.get("zotero_keys", {})

    all_edits: list[Edit] = []
    all_ambiguous: list[str] = []

    for section in sections:
        section_content = section["section_content"]

        cited_keys = extract_citation_keys_from_text(section_content)
        key_to_doi = {v: k for k, v in zotero_keys.items()}
        cited_dois = {key_to_doi.get(k) for k in cited_keys if k in key_to_doi}
        cited_summaries = {
            doi: state["paper_summaries"][doi]
            for doi in cited_dois if doi in state["paper_summaries"]
        }

        dynamic_max_chars = calculate_dynamic_char_budget(
            section_content=section_content,
            system_prompt=LOOP5_FACT_CHECK_SYSTEM,
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

        user_prompt = LOOP5_FACT_CHECK_USER.format(
            section_content=section_content,
            paper_summaries=f"{manifest_note}\n\n{paper_summaries_text}",
        )

        estimated_tokens = estimate_loop5_request_tokens(
            section_content=section_content,
            system_prompt=LOOP5_FACT_CHECK_SYSTEM,
            paper_summaries_text=f"{manifest_note}\n\n{paper_summaries_text}",
        )
        model_tier = select_model_tier_for_context(estimated_tokens)

        result = await get_structured_output(
            output_schema=DocumentEdits,
            user_prompt=user_prompt,
            system_prompt=LOOP5_FACT_CHECK_SYSTEM,
            tools=paper_tools,
            tier=model_tier,
            max_tokens=4096,
            max_tool_calls=5,
        )

        all_edits.extend(result.edits)
        all_ambiguous.extend(result.ambiguous_claims)

    logger.info(f"Loop 5: Fact check found {len(all_edits)} edits, {len(all_ambiguous)} ambiguous claims")
    return {
        "all_edits": all_edits,
        "ambiguous_claims": all_ambiguous,
    }


async def reference_check_node(state: Loop5State) -> dict[str, Any]:
    """Sequential reference checking across all sections with dynamic token budgeting."""
    sections = state["sections"]
    num_sections = len(sections)
    logger.info(f"Loop 5: Starting reference checking across {num_sections} sections")

    store_query = SupervisionStoreQuery(state["paper_summaries"])
    paper_tools = create_paper_tools(state["paper_summaries"], store_query)
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
        citation_keys_text = _format_citation_keys(cited_zotero_keys)

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
            tools=paper_tools,
            tier=model_tier,
            max_tokens=4096,
            max_tool_calls=5,
        )

        all_edits.extend(result.edits)
        all_todos.extend(result.unaddressed_todos)

    logger.info(f"Loop 5: Reference check complete, total edits: {len(all_edits)}, unaddressed TODOs: {len(all_todos)}")
    return {
        "all_edits": all_edits,
        "unaddressed_todos": all_todos,
    }


def validate_edits_node(state: Loop5State) -> dict[str, Any]:
    """Validate that all edit find strings exist and are unambiguous."""
    validation_result: EditValidationResult = validate_edits(
        state["current_review"], state["all_edits"]
    )

    human_items = []
    for idx, edit in enumerate(validation_result["invalid_edits"]):
        error_type = validation_result["errors"].get(str(idx), "unknown")
        human_items.append(
            f"Invalid edit ({error_type}): '{edit.find[:50]}...' -> '{edit.replace[:50]}...'"
        )

    logger.info(
        f"Loop 5: Validated edits - {len(validation_result['valid_edits'])} valid, "
        f"{len(validation_result['invalid_edits'])} invalid"
    )

    return {
        "valid_edits": validation_result["valid_edits"],
        "invalid_edits": validation_result["invalid_edits"],
        "human_review_items": human_items,
    }


async def apply_edits_node(state: Loop5State) -> dict[str, Any]:
    """Apply validated edits to the document with optional Zotero verification."""
    allowed_types = {"fact_correction", "citation_fix", "clarity"}
    filtered_edits = [
        edit for edit in state["valid_edits"] if edit.edit_type in allowed_types
    ]

    verify_zotero = state.get("verify_zotero", False)
    verified_keys = state.get("verified_citation_keys", set())
    zotero_keys = state.get("zotero_keys", {})
    corpus_keys = set(zotero_keys.values())

    updated_review = apply_edits(state["current_review"], filtered_edits)

    # Verify citations if enabled
    newly_verified: set[str] = set()
    if verify_zotero:
        zotero_client = ZoteroStore()
        try:
            valid_keys, invalid_keys = await validate_citations_against_zotero(
                text=updated_review,
                zotero_client=zotero_client,
                known_valid_keys=corpus_keys | verified_keys,
            )

            newly_verified = valid_keys - corpus_keys - verified_keys

            if invalid_keys:
                logger.warning(
                    f"Loop 5: Found {len(invalid_keys)} unverified citations, adding TODOs"
                )
                updated_review = strip_invalid_citations(
                    updated_review, invalid_keys, add_todo=True
                )
            else:
                logger.info("Loop 5: All citations verified in Zotero")

            if newly_verified:
                logger.info(f"Loop 5: Verified {len(newly_verified)} new citation keys")

        finally:
            await zotero_client.close()

    logger.info(f"Loop 5: Applied {len(filtered_edits)} edits to document")

    return {
        "current_review": updated_review,
        "verified_citation_keys": verified_keys | newly_verified,
    }


async def flag_issues_node(state: Loop5State) -> dict[str, Any]:
    """Collect ambiguous claims and unaddressed TODOs for human review."""
    human_items = state.get("human_review_items", []).copy()
    discarded_todos: list[str] = []

    for claim in state.get("ambiguous_claims", []):
        human_items.append(f"Ambiguous claim: {claim}")

    for todo in state.get("unaddressed_todos", []):
        human_items.append(f"Unaddressed TODO: {todo}")

    if state.get("verify_todos_enabled", True) and human_items:
        logger.info(f"Loop 5: Running TODO verification on {len(human_items)} items")
        try:
            verification_result = await verify_todos(
                todos=human_items,
                document=state["current_review"],
                topic=state.get("topic", ""),
                batch_size=30,
            )
            human_items = verification_result.keep
            discarded_todos = verification_result.discard
            logger.info(
                f"Loop 5: TODO verification kept {len(human_items)}, "
                f"discarded {len(discarded_todos)}"
            )
        except Exception as e:
            logger.error(f"Loop 5: TODO verification failed: {e}")

    return {
        "human_review_items": human_items,
        "discarded_todos": discarded_todos,
    }


def finalize_node(state: Loop5State) -> dict[str, Any]:
    """Mark loop as complete."""
    logger.info("Loop 5: Fact and reference checking complete")
    return {"is_complete": True}


def _format_paper_summaries(paper_summaries: dict[str, Any]) -> str:
    """Format paper summaries for prompt context."""
    if not paper_summaries:
        return "No paper summaries available."

    lines = []
    for doi, summary in paper_summaries.items():
        lines.append(f"DOI: {doi}")
        lines.append(f"Title: {summary.get('title', 'N/A')}")
        lines.append(f"Authors: {', '.join(summary.get('authors', []))}")
        lines.append(f"Year: {summary.get('year', 'N/A')}")
        lines.append(f"Summary: {summary.get('short_summary', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def _format_citation_keys(zotero_keys: dict[str, str]) -> str:
    """Format citation keys for reference checking."""
    if not zotero_keys:
        return "No citation keys available."

    lines = []
    for doi, key in zotero_keys.items():
        lines.append(f"[@{key}] -> {doi}")

    return "\n".join(lines)


def create_loop5_graph() -> StateGraph:
    """Create Loop 5 StateGraph for fact and reference checking."""
    graph = StateGraph(Loop5State)

    graph.add_node("split_sections", split_sections_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("reference_check", reference_check_node)
    graph.add_node("validate_edits", validate_edits_node)
    graph.add_node("apply_edits", apply_edits_node)
    graph.add_node("flag_issues", flag_issues_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "split_sections")
    graph.add_edge("split_sections", "fact_check")
    graph.add_edge("fact_check", "reference_check")
    graph.add_edge("reference_check", "validate_edits")
    graph.add_edge("validate_edits", "apply_edits")
    graph.add_edge("apply_edits", "flag_issues")
    graph.add_edge("flag_issues", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


async def run_loop5_standalone(
    review: str,
    paper_summaries: dict,
    zotero_keys: dict,
    max_iterations: int = 1,
    config: dict | None = None,
    topic: str = "",
    verify_todos_enabled: bool = True,
    verify_zotero: bool = False,
) -> dict:
    """Run Loop 5 as standalone operation for testing.

    Args:
        review: Literature review text to check
        paper_summaries: Paper summaries for fact checking
        zotero_keys: DOI -> Zotero key mapping for citation checking
        max_iterations: Maximum iterations (usually 1 for fact checking)
        config: Optional LangGraph config with run_id and run_name for tracing
        topic: Research topic for TODO verification context
        verify_todos_enabled: Whether to run TODO verification (default True)
        verify_zotero: If True, verify citations against Zotero programmatically

    Returns:
        Dict with current_review, human_review_items, discarded_todos,
        ambiguous_claims, unaddressed_todos, valid_edits, invalid_edits,
        and verified_citation_keys (if verify_zotero=True)
    """
    graph = create_loop5_graph()

    initial_state = Loop5State(
        current_review=review,
        paper_summaries=paper_summaries,
        zotero_keys=zotero_keys,
        sections=[],
        all_edits=[],
        valid_edits=[],
        invalid_edits=[],
        ambiguous_claims=[],
        unaddressed_todos=[],
        human_review_items=[],
        discarded_todos=[],
        iteration=0,
        max_iterations=max_iterations,
        is_complete=False,
        topic=topic,
        verify_todos_enabled=verify_todos_enabled,
        verify_zotero=verify_zotero,
        verified_citation_keys=set(),
    )

    if config:
        result = await graph.ainvoke(initial_state, config=config)
    else:
        result = await graph.ainvoke(initial_state)

    return {
        "current_review": result["current_review"],
        "human_review_items": result.get("human_review_items", []),
        "discarded_todos": result.get("discarded_todos", []),
        "ambiguous_claims": result.get("ambiguous_claims", []),
        "unaddressed_todos": result.get("unaddressed_todos", []),
        "valid_edits": result.get("valid_edits", []),
        "invalid_edits": result.get("invalid_edits", []),
        "verified_citation_keys": result.get("verified_citation_keys", set()),
    }
