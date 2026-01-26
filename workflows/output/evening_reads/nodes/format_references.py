"""Reference formatting node for evening_reads workflow.

Formats references for all 4 articles (1 overview + 3 deep-dives) using Zotero lookup.
"""

import asyncio
import logging
import re
from typing import Any

from core.stores.zotero import ZoteroStore
from core.stores.zotero.schemas import ZoteroItem

from ..state import (
    EveningReadsState,
    FinalOutput,
    FormattedReference,
)

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


def _extract_citations_from_text(text: str) -> list[str]:
    """Extract all [@KEY] citations from text."""
    matches = re.findall(CITATION_PATTERN, text)
    keys = set()
    for match in matches:
        for key in match.split(";"):
            key = key.strip().lstrip("@")
            if key:
                keys.add(key)
    return sorted(keys)


async def _lookup_citations(
    citation_keys: list[str],
) -> tuple[list[FormattedReference], list[str]]:
    """Look up citations in Zotero and format them.

    Returns:
        (formatted_references, missing_references)
    """
    if not citation_keys:
        return [], []

    formatted_refs: list[FormattedReference] = []
    missing_refs: list[str] = []

    try:
        async with ZoteroStore() as zotero:
            semaphore = asyncio.Semaphore(10)

            async def lookup_key(
                key: str,
            ) -> tuple[str, ZoteroItem | None, str | None]:
                async with semaphore:
                    try:
                        item = await zotero.get(key)
                        return key, item, None
                    except Exception as e:
                        return key, None, str(e)

            results = await asyncio.gather(
                *[lookup_key(k) for k in citation_keys],
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
        for key in citation_keys:
            missing_refs.append(key)
            formatted_refs.append(
                FormattedReference(
                    key=key,
                    citation_text=f"[Zotero unavailable: {key}]",
                    found_in_zotero=False,
                )
            )

    return formatted_refs, missing_refs


def _build_reference_section(
    formatted_refs: list[FormattedReference], citation_keys: list[str]
) -> str:
    """Build a reference section for the citations used in an article."""
    # Filter to only refs that are in this article's citations
    article_refs = [r for r in formatted_refs if r["key"] in citation_keys]

    # Sort alphabetically by citation text
    article_refs.sort(key=lambda r: r["citation_text"])

    found_refs = [r for r in article_refs if r["found_in_zotero"]]
    if not found_refs:
        return ""

    ref_section = "\n\n---\n\n## References\n\n"
    for ref in found_refs:
        ref_section += f"[@{ref['key']}] {ref['citation_text']}\n\n"

    return ref_section


async def format_references_node(state: EveningReadsState) -> dict[str, Any]:
    """Format references for all articles and produce final outputs.

    Looks up all unique citations across all drafts, then builds reference
    sections for each article.

    Returns:
        State update with formatted_references, missing_references, final_outputs
    """
    deep_dive_drafts = state.get("deep_dive_drafts", [])
    overview_draft = state.get("overview_draft")

    if not deep_dive_drafts and not overview_draft:
        return {
            "status": "failed",
            "errors": [{"node": "format_references", "error": "No drafts to format"}],
        }

    # Collect all unique citation keys across all articles
    all_citation_keys = set()

    for draft in deep_dive_drafts:
        all_citation_keys.update(_extract_citations_from_text(draft["content"]))

    if overview_draft:
        all_citation_keys.update(_extract_citations_from_text(overview_draft["content"]))

    logger.info(f"Looking up {len(all_citation_keys)} unique citations")

    # Look up all citations once
    formatted_refs, missing_refs = await _lookup_citations(sorted(all_citation_keys))

    found_count = len([r for r in formatted_refs if r["found_in_zotero"]])
    logger.info(f"Formatted {found_count}/{len(all_citation_keys)} references")

    # Build final outputs for each article
    final_outputs: list[FinalOutput] = []

    # Process deep-dives
    for draft in deep_dive_drafts:
        draft_citations = _extract_citations_from_text(draft["content"])
        ref_section = _build_reference_section(formatted_refs, draft_citations)
        final_content = draft["content"] + ref_section

        final_outputs.append(
            FinalOutput(
                id=draft["id"],
                title=draft["title"],
                content=final_content,
                word_count=len(final_content.split()),
            )
        )

    # Process overview
    if overview_draft:
        overview_citations = _extract_citations_from_text(overview_draft["content"])
        ref_section = _build_reference_section(formatted_refs, overview_citations)
        final_content = overview_draft["content"] + ref_section

        final_outputs.append(
            FinalOutput(
                id="overview",
                title=overview_draft["title"],
                content=final_content,
                word_count=len(final_content.split()),
            )
        )

    status = "success" if not missing_refs else "partial"

    return {
        "formatted_references": formatted_refs,
        "missing_references": missing_refs,
        "final_outputs": final_outputs,
        "status": status,
    }
