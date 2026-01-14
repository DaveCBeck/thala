"""Reference formatting node with Zotero API lookup."""

import asyncio
import logging
import re
from typing import Any

from core.stores.zotero import ZoteroStore
from core.stores.zotero.schemas import ZoteroItem
from ..state import SubstackReviewState, FormattedReference

logger = logging.getLogger(__name__)

CITATION_PATTERN = r"\[@([^\]]+)\]"


def _format_zotero_item_as_citation(item: ZoteroItem) -> str:
    """Format a ZoteroItem as an APA-style citation string."""
    fields = item.fields
    creators = item.creators

    # Format authors
    author_parts = []
    for c in creators[:3]:
        if c.get("lastName"):
            if c.get("firstName"):
                author_parts.append(f"{c['lastName']}, {c['firstName'][0]}.")
            else:
                author_parts.append(c["lastName"])
        elif c.get("name"):
            author_parts.append(c["name"])

    if len(creators) > 3:
        authors_str = ", ".join(author_parts) + ", et al."
    elif author_parts:
        authors_str = ", ".join(author_parts)
    else:
        authors_str = "Unknown"

    # Extract fields
    date_str = fields.get("date", "")
    year = date_str[:4] if date_str else "n.d."
    title = fields.get("title", "Untitled")
    publication = (
        fields.get("publicationTitle")
        or fields.get("journalAbbreviation")
        or fields.get("publisher")
        or ""
    )

    citation = f"{authors_str} ({year}). {title}."
    if publication:
        citation += f" *{publication}*."

    return citation


def _extract_citations_from_essay(text: str) -> list[str]:
    """Extract all [@KEY] citations from the essay text."""
    matches = re.findall(CITATION_PATTERN, text)
    keys = set()
    for match in matches:
        for key in match.split(";"):
            key = key.strip().lstrip("@")
            if key:
                keys.add(key)
    return sorted(keys)


async def format_references_node(state: SubstackReviewState) -> dict[str, Any]:
    """Look up citations in Zotero and format reference list.

    Extracts citations from the winning essay, looks up each in Zotero,
    and appends a formatted References section.
    """
    essay_drafts = state.get("essay_drafts", [])
    selected_angle = state.get("selected_angle")

    # Handle case where selection failed
    if not selected_angle:
        return {
            "status": "failed",
            "errors": [
                {"node": "format_references", "error": "No essay selected"}
            ],
        }

    # Get winning essay
    winning_essay = next(
        (e for e in essay_drafts if e["angle"] == selected_angle),
        None,
    )

    if not winning_essay:
        return {
            "status": "failed",
            "errors": [
                {"node": "format_references", "error": "Winning essay not found"}
            ],
        }

    # Extract citations from the winning essay
    essay_citation_keys = _extract_citations_from_essay(winning_essay["content"])
    logger.info(f"Found {len(essay_citation_keys)} citations in winning essay")

    if not essay_citation_keys:
        # No citations to format
        return {
            "formatted_references": [],
            "missing_references": [],
            "final_essay": winning_essay["content"],
            "status": "success",
        }

    # Look up each citation in Zotero
    formatted_refs: list[FormattedReference] = []
    missing_refs: list[str] = []

    try:
        async with ZoteroStore() as zotero:
            # Batch lookups with concurrency limit
            semaphore = asyncio.Semaphore(10)

            async def lookup_key(key: str) -> tuple[str, ZoteroItem | None, str | None]:
                async with semaphore:
                    try:
                        item = await zotero.get(key)
                        return key, item, None
                    except Exception as e:
                        return key, None, str(e)

            results = await asyncio.gather(
                *[lookup_key(k) for k in essay_citation_keys],
            )

            for key, item, error in results:
                if error:
                    logger.warning(f"Zotero lookup failed for {key}: {error}")
                    missing_refs.append(key)
                    formatted_refs.append(
                        FormattedReference(
                            key=key,
                            citation_text=f"[Reference not found: {key}]",
                            found_in_zotero=False,
                        )
                    )
                elif item is None:
                    logger.warning(f"Citation key not found in Zotero: {key}")
                    missing_refs.append(key)
                    formatted_refs.append(
                        FormattedReference(
                            key=key,
                            citation_text=f"[Reference not found: {key}]",
                            found_in_zotero=False,
                        )
                    )
                else:
                    citation_text = _format_zotero_item_as_citation(item)
                    formatted_refs.append(
                        FormattedReference(
                            key=key,
                            citation_text=citation_text,
                            found_in_zotero=True,
                        )
                    )

    except Exception as e:
        logger.error(f"Zotero connection failed: {e}")
        # Continue without references rather than failing
        for key in essay_citation_keys:
            missing_refs.append(key)
            formatted_refs.append(
                FormattedReference(
                    key=key,
                    citation_text=f"[Zotero unavailable: {key}]",
                    found_in_zotero=False,
                )
            )

    # Sort references alphabetically by citation text
    formatted_refs.sort(key=lambda r: r["citation_text"])

    # Build reference section
    found_refs = [r for r in formatted_refs if r["found_in_zotero"]]
    if found_refs:
        ref_section = "\n\n---\n\n## References\n\n"
        for ref in found_refs:
            ref_section += f"[@{ref['key']}] {ref['citation_text']}\n\n"
    else:
        ref_section = ""

    final_essay = winning_essay["content"] + ref_section

    found_count = len(found_refs)
    total_count = len(essay_citation_keys)
    logger.info(f"Formatted {found_count}/{total_count} references from Zotero")

    status = "success" if not missing_refs else "partial"

    return {
        "formatted_references": formatted_refs,
        "missing_references": missing_refs,
        "final_essay": final_essay,
        "status": status,
    }
