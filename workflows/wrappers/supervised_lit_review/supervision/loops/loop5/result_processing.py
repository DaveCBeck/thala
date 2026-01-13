"""Result processing and aggregation for Loop 5."""

import logging
from typing import Any

from core.stores.zotero import ZoteroStore
from ...types import Edit
from ...utils import (
    split_into_sections,
    validate_edits,
    apply_edits,
    EditValidationResult,
    extract_citation_keys_from_text,
    validate_citations_against_zotero,
)
from ..todo_verification import verify_todos

logger = logging.getLogger(__name__)


def split_sections_node(state: dict[str, Any]) -> dict[str, Any]:
    """Split document into sections for sequential checking."""
    sections = split_into_sections(state["current_review"])
    logger.debug(f"Split document into {len(sections)} sections for checking")
    return {"sections": sections}


def validate_edits_node(state: dict[str, Any]) -> dict[str, Any]:
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
        f"Edit validation complete: {len(validation_result['valid_edits'])} valid, "
        f"{len(validation_result['invalid_edits'])} invalid"
    )

    return {
        "valid_edits": validation_result["valid_edits"],
        "invalid_edits": validation_result["invalid_edits"],
        "human_review_items": human_items,
    }


async def apply_edits_node(state: dict[str, Any]) -> dict[str, Any]:
    """Apply validated edits and verify/fix invalid citations."""
    from .citation_resolution import resolve_invalid_citations

    allowed_types = {"fact_correction", "citation_fix", "clarity"}
    filtered_edits = [
        edit for edit in state["valid_edits"] if edit.edit_type in allowed_types
    ]

    updated_review = apply_edits(state["current_review"], filtered_edits)
    logger.info(f"Applied {len(filtered_edits)} edits to document")

    # Extract all citation keys from document
    all_keys = extract_citation_keys_from_text(updated_review)
    if not all_keys:
        logger.debug("No citations found in document")
        return {"current_review": updated_review}

    # Verify all citations against Zotero
    zotero_client = ZoteroStore()
    try:
        valid_keys, invalid_keys = await validate_citations_against_zotero(
            text=updated_review,
            zotero_client=zotero_client,
            known_valid_keys=set(),  # Verify all keys fresh
        )

        logger.info(f"Citation verification: {len(valid_keys)} valid, {len(invalid_keys)} invalid")

        if invalid_keys:
            # Use LLM to resolve invalid citations
            updated_review = await resolve_invalid_citations(
                document=updated_review,
                invalid_keys=invalid_keys,
                topic=state.get("topic", ""),
            )

    finally:
        await zotero_client.close()

    return {"current_review": updated_review}


def filter_ambiguous_claims(
    claims: list[str],
    false_positive_patterns: list[str],
) -> tuple[list[str], list[str]]:
    """Filter out corpus-gap and low-value ambiguous claims."""
    filtered = []
    discarded = []

    methodological_indicators = [
        "we used", "we employed", "we selected", "we chose",
        "this study used", "the approach", "methodology",
        "research design", "data collection", "sample size",
        "we analyzed", "we examined", "we investigated",
        "the authors", "researchers typically", "standard practice",
    ]

    for claim in claims:
        claim_lower = claim.lower()

        is_corpus_gap = any(pattern in claim_lower for pattern in false_positive_patterns)
        if is_corpus_gap:
            discarded.append(f"Pre-filtered (corpus gap): {claim}")
            logger.debug(f"Filtered corpus-gap claim: {claim[:80]}...")
            continue

        is_methodological = any(ind in claim_lower for ind in methodological_indicators)
        if is_methodological:
            discarded.append(f"Pre-filtered (methodological): {claim}")
            logger.debug(f"Filtered methodological claim: {claim[:80]}...")
            continue

        filtered.append(claim)

    if claims and len(claims) != len(filtered):
        logger.debug(
            f"Ambiguous claim filtering: {len(claims)} -> {len(filtered)} "
            f"({len(claims) - len(filtered)} filtered)"
        )

    return filtered, discarded


async def flag_issues_node(state: dict[str, Any]) -> dict[str, Any]:
    """Collect ambiguous claims and unaddressed TODOs for human review."""
    human_items = state.get("human_review_items", []).copy()
    discarded_todos: list[str] = []

    FALSE_POSITIVE_PATTERNS = [
        "unable to verify",
        "cannot verify",
        "could not verify",
        "no information found",
        "paper content is not available",
        "source documents unavailable",
        "cannot be verified against",
        "provided paper summaries",
        "not in provided papers",
        "provided summaries do not contain",
        "corpus does not contain",
        "not found in provided papers",
        "not in the reviewed literature",
        "not in our corpus",
        "sources unavailable",
        "retrieval system",
        "no papers in corpus",
        "paper not in corpus",
        "not in paper summaries",
        "not available in provided",
        "insufficient detail in",
        "limited information",
        "specific section numbers cannot be verified",
        "exact values depend on",
        "exact wording not verified",
        "precise statistics unavailable",
        "specific numbers not confirmed",
        "detailed data not in summaries",
    ]

    ambiguous_claims = state.get("ambiguous_claims", [])
    unaddressed_todos = state.get("unaddressed_todos", [])

    logger.debug(
        f"Collecting {len(ambiguous_claims)} ambiguous claims, "
        f"{len(unaddressed_todos)} unaddressed TODOs"
    )

    filtered_claims, claim_discards = filter_ambiguous_claims(
        claims=ambiguous_claims,
        false_positive_patterns=FALSE_POSITIVE_PATTERNS,
    )
    discarded_todos.extend(claim_discards)

    for claim in filtered_claims:
        human_items.append(f"Ambiguous claim: {claim}")

    for todo in unaddressed_todos:
        todo_lower = todo.lower()
        is_corpus_gap = any(pattern in todo_lower for pattern in FALSE_POSITIVE_PATTERNS)
        if is_corpus_gap:
            discarded_todos.append(f"Pre-filtered TODO (corpus gap): {todo}")
            logger.debug(f"Filtered corpus-gap TODO: {todo[:80]}...")
        else:
            human_items.append(f"Unaddressed TODO: {todo}")

    if human_items:
        logger.debug(f"Running TODO verification on {len(human_items)} items")
        try:
            verification_result = await verify_todos(
                todos=human_items,
                document=state["current_review"],
                topic=state.get("topic", ""),
                batch_size=30,
            )
            human_items = verification_result.keep
            discarded_todos.extend(verification_result.discard)
            logger.debug(
                f"TODO verification kept {len(human_items)}, "
                f"discarded {len(verification_result.discard)}"
            )
        except Exception as e:
            logger.error(f"TODO verification failed: {e}", exc_info=True)

    return {
        "human_review_items": human_items,
        "discarded_todos": discarded_todos,
    }


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    """Mark loop as complete and strip remaining TODO markers with WARNING logs."""
    import re

    current_review = state["current_review"]

    todo_pattern = r'<!-- TODO:.*?-->'
    todos = re.findall(todo_pattern, current_review, re.DOTALL)

    todos_stripped = 0
    if todos:
        logger.warning(f"Stripping {len(todos)} unresolved TODO markers")

        for todo in todos:
            idx = current_review.find(todo)
            if idx == -1:
                continue

            start = max(0, idx - 100)
            end = min(len(current_review), idx + len(todo) + 100)
            context = current_review[start:end]

            logger.warning(
                f"Stripping TODO:\n"
                f"  TODO: {todo[:80]}{'...' if len(todo) > 80 else ''}\n"
                f"  Context: ...{context.replace(todo, '[TODO]')}..."
            )

            todos_stripped += 1

        current_review = re.sub(todo_pattern, '', current_review, flags=re.DOTALL)
        current_review = re.sub(r'\n{3,}', '\n\n', current_review)

        logger.debug(f"Stripped {todos_stripped} TODO markers from final document")

    logger.info("Loop 5 complete")
    return {
        "current_review": current_review,
        "is_complete": True,
        "todos_stripped": todos_stripped,
    }
