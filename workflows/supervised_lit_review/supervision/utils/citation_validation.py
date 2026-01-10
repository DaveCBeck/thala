"""Pre-flight citation validation for Loop4 edits.

Validates that section edits only reference papers that have valid Zotero keys,
preventing citation drift where the LLM adds references to non-existent papers.
"""

import asyncio
import logging
import re
from typing import Set, Optional

from core.stores.zotero import ZoteroStore

logger = logging.getLogger(__name__)


# Citation key source tracking
CITATION_SOURCE_INITIAL = "initial_corpus"
CITATION_SOURCE_LOOP1 = "loop1_expansion"
CITATION_SOURCE_LOOP2 = "loop2_expansion"
CITATION_SOURCE_LOOP4 = "loop4_store_search"
CITATION_SOURCE_LOOP5 = "loop5_store_search"


def extract_citation_keys_from_text(text: str) -> Set[str]:
    """Extract all [@KEY] citation keys from text.

    Args:
        text: The text to search for citation keys

    Returns:
        Set of citation keys (without the @[] wrapper)

    Example:
        >>> extract_citation_keys_from_text("See [@smith2020] and [@jones2021]")
        {'smith2020', 'jones2021'}
    """
    pattern = r"\[@([^\]]+)\]"
    return set(re.findall(pattern, text))


async def verify_zotero_citation(key: str, zotero_client: ZoteroStore) -> bool:
    """Verify that a Zotero citation key exists.

    Args:
        key: The Zotero citation key to verify
        zotero_client: ZoteroStore instance for verification

    Returns:
        True if the key exists in Zotero, False otherwise
    """
    try:
        return await zotero_client.exists(key)
    except Exception as e:
        logger.warning(f"Zotero verification failed for key '{key}': {e}")
        return False


async def verify_zotero_citations_batch(
    keys: Set[str],
    zotero_client: ZoteroStore,
    concurrency: int = 10,
) -> dict[str, bool]:
    """Verify multiple Zotero citation keys in batch.

    Args:
        keys: Set of citation keys to verify
        zotero_client: ZoteroStore instance for verification
        concurrency: Max concurrent verification requests

    Returns:
        Dict mapping each key to its verification result (True/False)
    """
    if not keys:
        return {}

    semaphore = asyncio.Semaphore(concurrency)

    async def verify_with_semaphore(key: str) -> tuple[str, bool]:
        async with semaphore:
            exists = await verify_zotero_citation(key, zotero_client)
            return key, exists

    results = await asyncio.gather(*[verify_with_semaphore(k) for k in keys])
    return dict(results)


async def validate_citations_against_zotero(
    text: str,
    zotero_client: ZoteroStore,
    known_valid_keys: Optional[Set[str]] = None,
) -> tuple[Set[str], Set[str]]:
    """Validate all citations in text against Zotero.

    Args:
        text: Text containing [@KEY] citations
        zotero_client: ZoteroStore instance for verification
        known_valid_keys: Optional set of keys already known to be valid (skip verification)

    Returns:
        Tuple of (valid_keys, invalid_keys)
    """
    all_keys = extract_citation_keys_from_text(text)

    if not all_keys:
        return set(), set()

    known_valid = known_valid_keys or set()
    keys_to_verify = all_keys - known_valid

    if not keys_to_verify:
        return all_keys, set()

    verification_results = await verify_zotero_citations_batch(keys_to_verify, zotero_client)

    valid_keys = known_valid.copy()
    invalid_keys = set()

    for key, exists in verification_results.items():
        if exists:
            valid_keys.add(key)
        else:
            invalid_keys.add(key)

    return valid_keys, invalid_keys


def validate_edit_citations(
    original_section: str,
    edited_section: str,
    corpus_keys: Set[str],
    paper_summaries: Optional[dict] = None,
) -> tuple[bool, list[str]]:
    """Validate that edited section only cites papers with valid Zotero keys.

    This enforces the citation validation policy for Loop4 edits:
    - Citations with valid zotero keys in corpus_keys are allowed
    - Citations that match DOIs in paper_summaries (if provided) are allowed
    - All other citations are rejected

    Args:
        original_section: The original section content before editing
        edited_section: The section content after LLM editing
        corpus_keys: Set of valid zotero citation keys in the corpus
        paper_summaries: Optional dict of DOI -> PaperSummary for additional validation

    Returns:
        Tuple of (is_valid, list of invalid citations with reasons)
        is_valid is True if all citations can be validated
    """
    edited_citations = extract_citation_keys_from_text(edited_section)

    # Build additional key set from paper_summaries if available
    additional_keys: Set[str] = set()
    if paper_summaries:
        for doi, summary in paper_summaries.items():
            # Add zotero_key from summary if present
            if zotero_key := summary.get("zotero_key"):
                additional_keys.add(zotero_key)
            # Also add DOI-based fallback key
            doi_key = doi.replace("/", "_").replace(".", "_").upper()[:12]
            additional_keys.add(doi_key)

    all_valid_keys = corpus_keys | additional_keys

    # Validate citations
    invalid_citations = []
    for key in edited_citations:
        if key not in all_valid_keys:
            # Try looser matching - check if key is a substring or variant
            if not _is_plausible_citation_key(key, all_valid_keys):
                invalid_citations.append(f"{key} (not in corpus)")

    return len(invalid_citations) == 0, invalid_citations


async def validate_edit_citations_with_zotero(
    original_section: str,
    edited_section: str,
    corpus_keys: Set[str],
    zotero_client: ZoteroStore,
    paper_summaries: Optional[dict] = None,
) -> tuple[bool, list[str], Set[str]]:
    """Validate citations with programmatic Zotero verification.

    Extends validate_edit_citations by verifying new citations against Zotero.

    Args:
        original_section: The original section content before editing
        edited_section: The section content after LLM editing
        corpus_keys: Set of valid zotero citation keys in the corpus
        zotero_client: ZoteroStore instance for verification
        paper_summaries: Optional dict of DOI -> PaperSummary

    Returns:
        Tuple of (is_valid, invalid_citations_list, verified_keys)
        verified_keys contains all keys that passed Zotero verification
    """
    edited_citations = extract_citation_keys_from_text(edited_section)
    original_citations = extract_citation_keys_from_text(original_section)

    # Build additional key set from paper_summaries if available
    additional_keys: Set[str] = set()
    if paper_summaries:
        for doi, summary in paper_summaries.items():
            if zotero_key := summary.get("zotero_key"):
                additional_keys.add(zotero_key)
            doi_key = doi.replace("/", "_").replace(".", "_").upper()[:12]
            additional_keys.add(doi_key)

    all_corpus_keys = corpus_keys | additional_keys

    # Citations already in original section are assumed valid
    known_valid = original_citations & edited_citations

    # New citations need verification
    new_citations = edited_citations - original_citations - all_corpus_keys

    invalid_citations = []
    verified_keys = known_valid.copy()

    # Verify new citations against Zotero
    if new_citations:
        verification_results = await verify_zotero_citations_batch(
            new_citations, zotero_client
        )

        for key, exists in verification_results.items():
            if exists:
                verified_keys.add(key)
                logger.info(f"Citation [@{key}] verified in Zotero")
            else:
                invalid_citations.append(f"{key} (not found in Zotero)")
                logger.warning(f"Citation [@{key}] not found in Zotero")

    # Check citations not in corpus and not verified
    for key in edited_citations:
        if key not in all_corpus_keys and key not in verified_keys:
            if key not in [c.split(" ")[0] for c in invalid_citations]:
                if not _is_plausible_citation_key(key, all_corpus_keys):
                    invalid_citations.append(f"{key} (not in corpus)")

    is_valid = len(invalid_citations) == 0
    return is_valid, invalid_citations, verified_keys


def strip_invalid_citations(
    text: str,
    invalid_keys: Set[str],
    add_todo: bool = True,
) -> str:
    """Remove or mark invalid citations in text.

    Args:
        text: Text containing citations
        invalid_keys: Set of citation keys to remove/mark
        add_todo: If True, replace with TODO marker; if False, remove entirely

    Returns:
        Text with invalid citations handled
    """
    result = text
    for key in invalid_keys:
        pattern = rf"\[@{re.escape(key)}\]"
        if add_todo:
            replacement = f"<!-- TODO: Verify citation [@{key}] - not found in Zotero -->"
        else:
            replacement = ""
        result = re.sub(pattern, replacement, result)
    return result


def _is_plausible_citation_key(key: str, valid_keys: Set[str]) -> bool:
    """Check if a key could plausibly match a valid citation.

    Handles cases where the LLM generates slight variants of valid keys.

    Args:
        key: The citation key to check
        valid_keys: Set of known valid keys

    Returns:
        True if key is likely a valid variant, False otherwise
    """
    key_lower = key.lower()

    # Check for exact match (case-insensitive)
    for valid_key in valid_keys:
        if valid_key.lower() == key_lower:
            return True

    # Check for common variants (underscore vs hyphen, extra numbers)
    key_normalized = re.sub(r"[_\-]", "", key_lower)
    for valid_key in valid_keys:
        valid_normalized = re.sub(r"[_\-]", "", valid_key.lower())
        if key_normalized == valid_normalized:
            return True

    return False


def validate_corpus_zotero_keys(
    paper_summaries: dict,
    zotero_keys: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Validate that all corpus papers have Zotero keys and generate missing ones.

    This ensures that all papers in the corpus can be cited, either with their
    existing Zotero key or a generated fallback.

    Args:
        paper_summaries: Dict of DOI -> PaperSummary
        zotero_keys: Dict of DOI -> zotero citation key

    Returns:
        Tuple of (list of papers with missing keys, updated zotero_keys dict)
    """
    missing_keys: list[str] = []
    updated_keys = dict(zotero_keys)

    for doi in paper_summaries.keys():
        if doi not in zotero_keys:
            # Generate a fallback key from the DOI
            fallback_key = _generate_fallback_key(doi, paper_summaries.get(doi, {}))
            updated_keys[doi] = fallback_key
            missing_keys.append(doi)
            logger.warning(
                f"Generated fallback Zotero key '{fallback_key}' for paper {doi}"
            )

    if missing_keys:
        logger.info(
            f"Generated {len(missing_keys)} fallback Zotero keys for corpus papers"
        )

    return missing_keys, updated_keys


def _generate_fallback_key(doi: str, summary: dict) -> str:
    """Generate a fallback Zotero citation key from DOI or metadata.

    Uses author surname + year format when possible, otherwise DOI-based.

    Args:
        doi: The paper DOI
        summary: Paper summary with metadata

    Returns:
        Generated citation key
    """
    # Try to use author + year format
    authors = summary.get("authors", [])
    year = summary.get("year")

    if authors and year:
        # Get first author's surname
        first_author = authors[0]
        if "," in first_author:
            surname = first_author.split(",")[0].strip()
        else:
            surname = first_author.split()[-1] if first_author.split() else ""

        if surname:
            # Clean surname (remove non-alphanumeric)
            surname_clean = re.sub(r"[^a-zA-Z]", "", surname).upper()[:8]
            return f"{surname_clean}{year}"

    # Fallback: generate from DOI
    # Take last segment of DOI, clean it up
    doi_suffix = doi.split("/")[-1] if "/" in doi else doi
    clean_suffix = re.sub(r"[^a-zA-Z0-9]", "", doi_suffix).upper()[:12]
    return clean_suffix if clean_suffix else "UNKNOWN"


def check_section_growth(
    original: str,
    edited: str,
    tolerance: float = 0.20,
) -> tuple[bool, float]:
    """Check if edited section is within tolerance of original length.

    Loop4 enforces a +/-20% word count limit to prevent excessive expansion
    or reduction of section content.

    Args:
        original: Original section content
        edited: Edited section content
        tolerance: Maximum allowed growth/shrinkage ratio (default 0.20 = +/-20%)

    Returns:
        Tuple of (is_within_tolerance, growth_ratio)
        growth_ratio is positive for expansion, negative for reduction
    """
    original_words = len(original.split())
    edited_words = len(edited.split())

    if original_words == 0:
        return True, 0.0

    growth_ratio = (edited_words - original_words) / original_words
    is_within_tolerance = abs(growth_ratio) <= tolerance
    return is_within_tolerance, growth_ratio
