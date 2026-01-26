"""
Update Zotero item with extracted data node.

Merges OpenAlex baseline metadata with document-extracted metadata,
validates fields, and updates the Zotero item.
"""

import logging
from typing import Any

from langsmith import traceable

from core.stores.zotero import ZoteroCreator, ZoteroItemUpdate
from langchain_tools.base import get_store_manager
from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.metadata_utils import (
    merge_metadata_with_baseline,
    parse_author_name,
    validate_year,
)

logger = logging.getLogger(__name__)


def _get_baseline_metadata(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Extract baseline metadata from OpenAlex/extra_metadata.

    OpenAlex data is generally reliable and serves as the trusted baseline.
    """
    doc_input = state.get("input", {})
    extra_metadata = doc_input.get("extra_metadata", {})

    # Convert OpenAlex authors format if needed
    # OpenAlex uses list of dicts with 'name' key
    baseline_authors = extra_metadata.get("authors", [])
    if baseline_authors and isinstance(baseline_authors[0], dict):
        baseline_authors = [a.get("name", "") for a in baseline_authors if a.get("name")]

    return {
        "authors": baseline_authors,
        "date": extra_metadata.get("publication_date") or extra_metadata.get("date"),
        "year": extra_metadata.get("year"),
        "title": extra_metadata.get("title") or doc_input.get("title"),
        "publisher": extra_metadata.get("publisher"),
    }


@traceable(run_type="chain", name="UpdateZotero")
async def update_zotero(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Update Zotero item with summary and validated metadata.

    Merge strategy (fill gaps only):
    - OpenAlex baseline is trusted for dates and authors
    - Document extraction fills in missing fields
    - All dates validated to ensure valid years
    - Author names properly parsed into firstName/lastName

    Sets abstractNote to short_summary, removes "pending" tag and adds "processed" tag.

    Returns current_status.
    """
    try:
        zotero_key = state.get("zotero_key")
        if not zotero_key:
            logger.warning("No zotero_key in state, skipping Zotero update")
            return {"current_status": "update_zotero_skipped"}

        short_summary = state.get("short_summary")
        metadata_updates = state.get("metadata_updates", {})

        # Get baseline from OpenAlex
        baseline = _get_baseline_metadata(state)

        # Merge extracted metadata with baseline (OpenAlex preferred for dates/authors)
        merged = merge_metadata_with_baseline(baseline, metadata_updates)

        store_manager = get_store_manager()

        # Build update payload
        update = ZoteroItemUpdate()

        # Update fields
        fields: dict[str, Any] = {}
        if short_summary:
            fields["abstractNote"] = short_summary
        if "title" in merged:
            fields["title"] = merged["title"]

        # Validate and set date - prefer year field, fall back to date
        date_to_validate = merged.get("year") or merged.get("date")
        if date_to_validate:
            validated_year = validate_year(date_to_validate)
            if validated_year:
                fields["date"] = validated_year
            else:
                logger.warning(f"Invalid year value, not updating: {date_to_validate}")

        if "publisher" in merged:
            fields["publisher"] = merged["publisher"]
        if "isbn" in merged:
            fields["ISBN"] = merged["isbn"]

        if fields:
            update.fields = fields

        # Update authors if present - use proper name parsing
        if "authors" in merged and merged["authors"]:
            creators = []
            for author_name in merged["authors"]:
                if not author_name or not author_name.strip():
                    continue
                parsed = parse_author_name(author_name)
                creators.append(
                    ZoteroCreator(
                        firstName=parsed.firstName,
                        lastName=parsed.lastName,
                        name=parsed.name,
                        creatorType="author",
                    )
                )
            if creators:
                update.creators = creators

        # Update tags: remove "pending", add "processed"
        # First, get current item to preserve other tags
        current_item = await store_manager.zotero.get(zotero_key)
        if current_item:
            current_tags = [
                tag.get("tag", "") for tag in current_item.tags if isinstance(tag, dict)
            ]
            # Remove "pending", add "processed"
            new_tags = [t for t in current_tags if t.lower() != "pending"]
            if "processed" not in [t.lower() for t in new_tags]:
                new_tags.append("processed")
            update.tags = new_tags
        else:
            update.tags = ["processed"]

        # Perform update
        success = await store_manager.zotero.update(zotero_key, update)

        if success:
            logger.info(f"Updated Zotero item: {zotero_key}")
            return {"current_status": "update_zotero_complete"}
        else:
            logger.warning(f"Zotero item not found: {zotero_key}")
            return {"current_status": "update_zotero_not_found"}

    except Exception as e:
        logger.error(f"Failed to update Zotero: {e}")
        return {
            "current_status": "update_zotero_failed",
            "errors": [{"node": "update_zotero", "error": str(e)}],
        }
