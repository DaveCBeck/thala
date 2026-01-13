"""Pre-flight citation validation for Loop4 edits.

Validates that section edits only reference papers that have valid Zotero keys,
preventing citation drift where the LLM adds references to non-existent papers.
"""

from .parsers import (
    extract_citation_keys_from_text,
    strip_invalid_citations,
    check_section_growth,
)
from .types import (
    CITATION_SOURCE_INITIAL,
    CITATION_SOURCE_LOOP1,
    CITATION_SOURCE_LOOP2,
    CITATION_SOURCE_LOOP4,
    CITATION_SOURCE_LOOP5,
)
from .validator import (
    validate_edit_citations,
    validate_edit_citations_with_zotero,
)
from .zotero import (
    verify_zotero_citation,
    verify_zotero_citations_batch,
    validate_citations_against_zotero,
    validate_corpus_zotero_keys,
)

__all__ = [
    "extract_citation_keys_from_text",
    "strip_invalid_citations",
    "check_section_growth",
    "CITATION_SOURCE_INITIAL",
    "CITATION_SOURCE_LOOP1",
    "CITATION_SOURCE_LOOP2",
    "CITATION_SOURCE_LOOP4",
    "CITATION_SOURCE_LOOP5",
    "validate_edit_citations",
    "validate_edit_citations_with_zotero",
    "verify_zotero_citation",
    "verify_zotero_citations_batch",
    "validate_citations_against_zotero",
    "validate_corpus_zotero_keys",
]
