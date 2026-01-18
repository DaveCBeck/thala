"""Reference-check section worker for fact-check workflow."""

import logging
import re
from typing import Any

from langgraph.types import Send

from workflows.enhance.editing.document_model import DocumentModel
from workflows.enhance.fact_check.schemas import (
    ReferenceCheckResult,
)
from workflows.enhance.fact_check.prompts import (
    REFERENCE_CHECK_SYSTEM,
    REFERENCE_CHECK_USER,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)

# Zotero citation pattern: [@8ALPHANUMERIC]
ZOTERO_CITATION_PATTERN = re.compile(r'\[@([A-Za-z0-9]{8})\]')

# Module-level citation cache (cleared between workflow runs)
_citation_validation_cache: dict[str, dict] = {}


def extract_section_citations(text: str) -> list[str]:
    """Extract citation keys from section text."""
    matches = ZOTERO_CITATION_PATTERN.findall(text)
    seen = set()
    unique = []
    for key in matches:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


async def pre_validate_citations(state: dict) -> dict[str, Any]:
    """Pre-validate all unique citations in the document.

    This caches citation existence checks to avoid redundant validation
    when the same citation appears in multiple sections.

    Returns:
        State update with citation_cache containing validation results.
    """
    global _citation_validation_cache
    _citation_validation_cache.clear()  # Clear cache for new workflow run

    document_model_dict = state.get("updated_document_model", state.get("document_model"))
    if not document_model_dict:
        return {"citation_cache": {}}

    document_model = DocumentModel.from_dict(document_model_dict)

    # Collect all unique citations across all sections
    all_citations = set()
    all_sections = document_model.get_all_sections()
    for section in all_sections:
        content = document_model.get_section_content(section.section_id, include_subsections=False)
        citations = extract_section_citations(content)
        all_citations.update(citations)

    if not all_citations:
        return {"citation_cache": {}}

    logger.debug(f"Pre-validating {len(all_citations)} unique citations")

    # Validate each citation's existence
    from langchain_tools import get_paper_content

    validated = {}
    for citation_key in all_citations:
        try:
            # Just check if paper exists and get basic info
            content = await get_paper_content.ainvoke({"zotero_key": citation_key})
            exists = bool(content and "not found" not in content.lower())
            validated[citation_key] = {
                "exists": exists,
                "content_preview": content[:500] if content else "",
            }
        except Exception as e:
            logger.debug(f"Citation {citation_key} validation failed: {e}")
            validated[citation_key] = {"exists": False, "content_preview": ""}

    # Cache results for section workers
    _citation_validation_cache.update(validated)

    exists_count = sum(1 for v in validated.values() if v.get("exists"))
    logger.info(
        f"Pre-validated {len(validated)} citations: "
        f"{exists_count} exist, {len(validated) - exists_count} not found"
    )

    return {"citation_cache": validated}


def get_cached_citation(citation_key: str) -> dict | None:
    """Get cached validation result for a citation."""
    return _citation_validation_cache.get(citation_key)


def route_to_reference_check_sections(state: dict) -> list[Send] | str:
    """Route to reference-check workers for sections with citations.

    Uses pre-validated citation cache to skip sections where all citations
    are already known to be invalid.

    Returns list of Send objects for parallel reference-checking,
    or "apply_verified_edits" if no sections have citations.
    """
    document_model_dict = state.get("updated_document_model", state.get("document_model"))
    if not document_model_dict:
        return "apply_verified_edits"

    document_model = DocumentModel.from_dict(document_model_dict)
    quality_settings = state.get("quality_settings", {})
    citation_cache = state.get("citation_cache", {})
    max_tool_calls_base = quality_settings.get("reference_check_max_tool_calls", 5)

    # Get leaf sections only (no subsections) to avoid duplicating content
    all_sections = document_model.get_all_sections()
    leaf_sections = [s for s in all_sections if not s.subsections]
    sections_to_check = []

    for section in leaf_sections:
        # Get only this section's content, not subsections
        content = document_model.get_section_content(section.section_id, include_subsections=False)
        citations = extract_section_citations(content)
        if citations:
            # Filter out citations that are already known to not exist
            # (still need to check claim support for ones that do exist)
            valid_citations = [
                c for c in citations
                if citation_cache.get(c, {}).get("exists", True)  # Default to checking if not cached
            ]
            if valid_citations or not citation_cache:  # Always check if no cache yet
                sections_to_check.append((section, citations))

    if not sections_to_check:
        return "apply_verified_edits"

    # Build Send objects for parallel reference-checking
    sends = []
    for section, citations in sections_to_check:
        # Limit tool calls based on citation count + quality setting
        max_tool_calls = min(max_tool_calls_base + len(citations), 20)

        sends.append(
            Send(
                "reference_check_section",
                {
                    "section_id": section.section_id,
                    "section_content": document_model.get_section_content(
                        section.section_id, include_subsections=False
                    ),
                    "section_heading": section.heading,
                    "citations": citations,
                    "topic": state["input"]["topic"],
                    "confidence_threshold": quality_settings.get("verify_confidence_threshold", 0.75),
                    "max_tool_calls": max_tool_calls,
                    "citation_cache": citation_cache,
                },
            )
        )

    logger.info(f"Routing to reference-check {len(sends)} sections with citations")
    return sends


async def reference_check_section_worker(state: dict) -> dict[str, Any]:
    """Check citation validity in a single section.

    This worker:
    1. Validates each citation key exists in Zotero/corpus
    2. Checks if cited paper supports the claim it's attached to
    3. Suggests edits for invalid or unsupported citations
    4. Returns validation results with suggested edits
    """
    section_id = state["section_id"]
    section_content = state["section_content"]
    section_heading = state["section_heading"]
    citations = state["citations"]
    topic = state["topic"]
    confidence_threshold = state.get("confidence_threshold", 0.75)
    max_tool_calls = state.get("max_tool_calls", len(citations) + 5)
    citation_cache = state.get("citation_cache", {})

    # Identify which citations need full checking vs already validated
    cached_invalid = [c for c in citations if not citation_cache.get(c, {}).get("exists", True)]
    citations_to_check = [c for c in citations if c not in cached_invalid]

    if cached_invalid:
        logger.debug(
            f"Skipping {len(cached_invalid)} cached-invalid citations in '{section_heading}'"
        )

    logger.debug(
        f"Reference-checking section '{section_heading}' "
        f"({len(citations_to_check)} citations to check, {len(cached_invalid)} cached-invalid)"
    )

    # Get paper tools
    from langchain_tools import search_papers, get_paper_content
    tools = [search_papers, get_paper_content]

    # Build prompt with cache context
    cache_context = ""
    if citation_cache:
        cached_info = []
        for c in citations_to_check:
            cache_entry = citation_cache.get(c, {})
            if cache_entry.get("exists"):
                preview = cache_entry.get("content_preview", "")[:200]
                cached_info.append(f"[@{c}]: exists, preview: {preview}...")
        if cached_info:
            cache_context = "\n\nPRE-VALIDATED CITATIONS:\n" + "\n".join(cached_info)

    user_prompt = REFERENCE_CHECK_USER.format(
        section_heading=section_heading,
        section_content=section_content,
        citations=", ".join(f"[@{c}]" for c in citations_to_check) if citations_to_check else "(none)",
        topic=topic,
        confidence_threshold=confidence_threshold,
    ) + cache_context

    try:
        result = await get_structured_output(
            output_schema=ReferenceCheckResult,
            user_prompt=user_prompt,
            system_prompt=REFERENCE_CHECK_SYSTEM,
            tier=ModelTier.SONNET,
            tools=tools,
            max_tokens=4000,
            max_tool_calls=max_tool_calls,
            use_json_schema_method=True,
        )

        # Add cached-invalid citations to invalid list
        result.invalid_citations = list(set(result.invalid_citations) | set(cached_invalid))

        # Override section_id and citations from result to match input
        result.section_id = section_id
        result.citations_found = citations

        # Log invalid and unsupported citations at INFO level
        if result.invalid_citations:
            for citation in result.invalid_citations:
                logger.info(
                    f"Invalid citation in '{section_heading}': [@{citation}] - not found in corpus"
                )

        if result.unsupported_citations:
            for citation in result.unsupported_citations:
                logger.info(
                    f"Unsupported citation in '{section_heading}': [@{citation}] - "
                    "paper content doesn't support the claim"
                )

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
            logger.debug(
                f"Skipping {len(low_confidence_edits)} low-confidence citation edits "
                f"in '{section_heading}'"
            )

        result.suggested_edits = valid_edits

        logger.info(
            f"Reference-checked section '{section_heading}': "
            f"{len(citations)} citations validated, "
            f"{len(result.invalid_citations)} invalid, "
            f"{len(result.unsupported_citations)} unsupported, "
            f"{len(valid_edits)} edits suggested"
        )

        return {
            "reference_check_results": [result.model_dump()],
            "pending_edits": [e.model_dump() for e in valid_edits],
        }

    except Exception as e:
        logger.error(f"Reference-check failed for section '{section_heading}': {e}")
        return {
            "reference_check_results": [
                ReferenceCheckResult(
                    section_id=section_id,
                    citations_found=citations,
                    validations=[],
                    invalid_citations=[],
                    unsupported_citations=[],
                    suggested_edits=[],
                ).model_dump()
            ],
            "errors": [{"node": "reference_check_section", "error": str(e)}],
        }


async def assemble_reference_checks_node(state: dict) -> dict[str, Any]:
    """Assemble reference-check results from parallel workers.

    Collects all pending edits and issues.
    """
    reference_check_results = state.get("reference_check_results", [])

    total_citations = sum(len(r.get("citations_found", [])) for r in reference_check_results)
    total_invalid = sum(len(r.get("invalid_citations", [])) for r in reference_check_results)
    total_unsupported = sum(len(r.get("unsupported_citations", [])) for r in reference_check_results)
    total_edits = sum(len(r.get("suggested_edits", [])) for r in reference_check_results)

    logger.info(
        f"Assembled reference-checks: {total_citations} citations validated, "
        f"{total_invalid} invalid, {total_unsupported} unsupported, "
        f"{total_edits} edits pending"
    )

    # Collect unresolved items (invalid/unsupported citations that couldn't be fixed)
    unresolved_items = state.get("unresolved_items", [])

    for result in reference_check_results:
        section_id = result.get("section_id")

        # Invalid citations without suggested edits are unresolved
        suggested_edit_citations = set()
        for e in result.get("suggested_edits", []):
            src_ref = e.get("source_reference") or ""
            suggested_edit_citations.add(src_ref.replace("[@", "").replace("]", ""))

        for citation in result.get("invalid_citations", []):
            if citation not in suggested_edit_citations:
                unresolved_items.append({
                    "source": "reference_check",
                    "section_id": section_id,
                    "issue": f"Invalid citation [@{citation}] - not found and no alternative available",
                })

        for citation in result.get("unsupported_citations", []):
            if citation not in suggested_edit_citations:
                unresolved_items.append({
                    "source": "reference_check",
                    "section_id": section_id,
                    "issue": f"Unsupported citation [@{citation}] - paper doesn't support the claim",
                })

    return {
        "unresolved_items": unresolved_items,
    }
