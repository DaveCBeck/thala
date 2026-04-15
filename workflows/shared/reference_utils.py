"""Split, augment, and reattach the trailing references block of a review.

Integration-phase LLM calls (Loop 1 supervision, Loop 2 literature, the
lit_review_web_augmented combine phase) should operate on prose only —
the references section is managed here so the model never re-types it.

Supported trailing-section headings (case-insensitive, matches at the end
of the document):

- ``## References``
- ``## Sources`` (including ``## Sources from Research``)
- ``## Bibliography``

An optional ``---`` horizontal rule immediately before the heading is
captured with the block so body/heading separation is clean on reattach.
"""

from __future__ import annotations

import re
from typing import Any

_REFERENCES_HEADING = re.compile(
    r"(\n+(?:---+\s*\n+)?)(## (?:References|Sources|Bibliography)\b.*\Z)",
    re.DOTALL | re.IGNORECASE,
)


def split_references(review: str) -> tuple[str, str]:
    """Split a markdown review into (body, trailing_references_block).

    ``body`` is trimmed of trailing whitespace and ends with a single
    newline. The block retains its leading separator and heading verbatim.
    Returns ``(review_stripped, "")`` if no recognisable trailing
    references section is present.
    """
    match = _REFERENCES_HEADING.search(review)
    if not match:
        return review.rstrip() + "\n", ""
    body = review[: match.start()].rstrip() + "\n"
    refs_block = match.group(1) + match.group(2)
    return body, refs_block


def format_reference_line(
    summary: dict[str, Any] | None,
    zotero_key: str,
) -> str:
    """Render one ``[@KEY] Authors (Year). Title. Venue.`` line.

    Mirrors the format used by
    ``workflows/research/academic_lit_review/synthesis/nodes/citation_nodes.py``
    so references added here are indistinguishable from those produced
    by the original Phase 1 synthesis.
    """
    summary = summary or {}
    authors = summary.get("authors") or []
    if len(authors) > 3:
        author_str = ", ".join(authors[:3]) + " et al."
    elif authors:
        author_str = ", ".join(authors)
    else:
        author_str = "Unknown authors"
    year = summary.get("year", "n.d.")
    title = summary.get("title", "Untitled")
    venue = summary.get("venue") or "Unknown venue"
    return f"[@{zotero_key}] {author_str} ({year}). {title}. {venue}.\n\n"


def append_new_references(
    refs_block: str,
    new_dois: list[str],
    paper_summaries: dict[str, dict[str, Any]],
    zotero_keys: dict[str, str],
) -> str:
    """Append reference entries for newly added papers.

    Entries whose ``[@KEY]`` already appears in ``refs_block`` are skipped
    so repeated integration calls don't produce duplicates. If
    ``refs_block`` is empty but ``new_dois`` contains valid entries, a
    fresh ``## References`` section is created.
    """
    if not new_dois:
        return refs_block

    new_lines: list[str] = []
    for doi in new_dois:
        key = zotero_keys.get(doi)
        if not key:
            continue
        if refs_block and f"[@{key}]" in refs_block:
            continue
        new_lines.append(format_reference_line(paper_summaries.get(doi), key))

    if not new_lines:
        return refs_block

    if not refs_block:
        return "\n\n## References\n\n" + "".join(new_lines).rstrip() + "\n"

    return refs_block.rstrip() + "\n\n" + "".join(new_lines).rstrip() + "\n"


def reattach(body: str, refs_block: str) -> str:
    """Reattach a references block to a body.

    If the LLM-returned body already contains a trailing references
    heading (the prompt told it not to, but integrators occasionally
    ignore that), keep what it returned and discard the external block —
    better to trust a self-consistent document than to produce two
    references sections.
    """
    if _REFERENCES_HEADING.search(body):
        return body
    if not refs_block:
        return body
    return body.rstrip() + refs_block


def merge_references_blocks(
    academic_refs: str,
    web_refs: str,
) -> str:
    """Merge academic-style and web-style references into one block.

    - If only one is present, return it unchanged.
    - If both are present, return a combined ``## References`` block with
      ``### Academic References`` and ``### Web Sources`` subsections.
    """
    if not academic_refs and not web_refs:
        return ""
    if academic_refs and not web_refs:
        return academic_refs
    if not academic_refs and web_refs:
        return web_refs

    def _entries_only(block: str) -> str:
        # Strip leading whitespace/separator and the `## Heading` line.
        stripped = block.lstrip("\n")
        stripped = re.sub(r"^---+\s*\n+", "", stripped)
        stripped = re.sub(r"^## [^\n]*\n+", "", stripped)
        return stripped.strip()

    academic_entries = _entries_only(academic_refs)
    web_entries = _entries_only(web_refs)

    parts = ["\n\n## References\n\n"]
    if academic_entries:
        parts.append("### Academic References\n\n")
        parts.append(academic_entries)
        parts.append("\n\n")
    if web_entries:
        parts.append("### Web Sources\n\n")
        parts.append(web_entries)
        parts.append("\n")
    return "".join(parts)


__all__ = [
    "split_references",
    "format_reference_line",
    "append_new_references",
    "reattach",
    "merge_references_blocks",
]
