"""Main citation validation logic for Loop4 edits."""

import logging
from typing import Set, Optional

from core.stores.zotero import ZoteroStore

from .parsers import extract_citation_keys_from_text, is_plausible_citation_key
from .zotero import verify_zotero_citations_batch

logger = logging.getLogger(__name__)


def validate_edit_citations(
    original_section: str,
    edited_section: str,
    corpus_keys: Set[str],
    paper_summaries: Optional[dict] = None,
) -> tuple[bool, list[str]]:
    """Validate that edited section only cites papers with valid Zotero keys."""
    edited_citations = extract_citation_keys_from_text(edited_section)

    additional_keys: Set[str] = set()
    if paper_summaries:
        for doi, summary in paper_summaries.items():
            if zotero_key := summary.get("zotero_key"):
                additional_keys.add(zotero_key)
            doi_key = doi.replace("/", "_").replace(".", "_").upper()[:12]
            additional_keys.add(doi_key)

    all_valid_keys = corpus_keys | additional_keys

    invalid_citations = []
    for key in edited_citations:
        if key not in all_valid_keys:
            if not is_plausible_citation_key(key, all_valid_keys):
                invalid_citations.append(f"{key} (not in corpus)")

    return len(invalid_citations) == 0, invalid_citations


async def validate_edit_citations_with_zotero(
    original_section: str,
    edited_section: str,
    corpus_keys: Set[str],
    zotero_client: ZoteroStore,
    paper_summaries: Optional[dict] = None,
    verify_all: bool = False,
) -> tuple[bool, list[str], Set[str]]:
    """Validate citations with programmatic Zotero verification.

    Args:
        original_section: Original section text before editing
        edited_section: Edited section text to validate
        corpus_keys: Set of known valid citation keys from corpus
        zotero_client: Zotero client for verification
        paper_summaries: Optional paper summaries with additional keys
        verify_all: If True, verify ALL citations against Zotero (making Zotero
            the source of truth). If False, only verify new citations not in corpus.
    """
    edited_citations = extract_citation_keys_from_text(edited_section)
    original_citations = extract_citation_keys_from_text(original_section)

    additional_keys: Set[str] = set()
    if paper_summaries:
        for doi, summary in paper_summaries.items():
            if zotero_key := summary.get("zotero_key"):
                additional_keys.add(zotero_key)
            doi_key = doi.replace("/", "_").replace(".", "_").upper()[:12]
            additional_keys.add(doi_key)

    all_corpus_keys = corpus_keys | additional_keys

    invalid_citations = []
    verified_keys: Set[str] = set()

    if verify_all:
        # Verify ALL citations against Zotero - Zotero is source of truth
        # If a citation exists in Zotero, it's valid regardless of corpus_keys
        citations_to_verify = edited_citations
        logger.info(f"Verifying all {len(citations_to_verify)} citations against Zotero")

        if citations_to_verify:
            verification_results = await verify_zotero_citations_batch(
                citations_to_verify, zotero_client
            )

            for key, exists in verification_results.items():
                if exists:
                    verified_keys.add(key)
                    logger.debug(f"Citation [@{key}] verified in Zotero")
                else:
                    # Only mark as invalid if not in corpus_keys either
                    if key not in all_corpus_keys:
                        invalid_citations.append(f"{key} (not found in Zotero)")
                        logger.warning(f"Citation [@{key}] not found in Zotero or corpus")
                    else:
                        # In corpus but not in Zotero - trust the corpus
                        verified_keys.add(key)
                        logger.debug(f"Citation [@{key}] in corpus but not in Zotero")

        logger.info(f"Zotero verification: {len(verified_keys)} valid, {len(invalid_citations)} invalid")

    else:
        # Original behavior: only verify NEW citations not in corpus
        known_valid = original_citations & edited_citations
        verified_keys = known_valid.copy()

        new_citations = edited_citations - original_citations - all_corpus_keys

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

        # Check remaining citations against corpus
        for key in edited_citations:
            if key not in all_corpus_keys and key not in verified_keys:
                if key not in [c.split(" ")[0] for c in invalid_citations]:
                    if not is_plausible_citation_key(key, all_corpus_keys):
                        invalid_citations.append(f"{key} (not in corpus)")

    is_valid = len(invalid_citations) == 0
    return is_valid, invalid_citations, verified_keys
