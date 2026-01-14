"""Zotero item creation."""

import logging
from typing import Optional

from core.stores.zotero import ZoteroItemCreate, ZoteroCreator

logger = logging.getLogger(__name__)

TYPE_MAPPING = {
    "webpage": "webpage",
    "journalArticle": "journalArticle",
    "blogPost": "blogPost",
    "report": "report",
    "newspaperArticle": "newspaperArticle",
    "magazineArticle": "magazineArticle",
    "book": "book",
    "bookSection": "bookSection",
    "conferencePaper": "conferencePaper",
    "thesis": "thesis",
    "document": "document",
}


async def _create_zotero_item(
    url: str,
    metadata: dict,
    store_manager,
) -> Optional[str]:
    """Create Zotero item and return the key."""
    # Build creators list
    creators = []
    for author in metadata.get("authors") or []:
        if isinstance(author, str) and author.strip():
            parts = author.strip().split(" ", 1)
            if len(parts) == 2:
                creators.append(
                    ZoteroCreator(
                        firstName=parts[0],
                        lastName=parts[1],
                        creatorType="author",
                    )
                )
            else:
                creators.append(
                    ZoteroCreator(
                        name=author,
                        creatorType="author",
                    )
                )

    # Map item types to Zotero types
    item_type = metadata.get("item_type", "webpage")
    zotero_item_type = TYPE_MAPPING.get(item_type, "webpage")

    # Build fields
    fields = {
        "title": metadata.get("title") or url,
        "url": url,
    }

    if metadata.get("date"):
        fields["date"] = metadata["date"]
    if metadata.get("publication_title"):
        if zotero_item_type == "webpage":
            fields["websiteTitle"] = metadata["publication_title"]
        else:
            fields["publicationTitle"] = metadata["publication_title"]
    if metadata.get("abstract"):
        fields["abstractNote"] = metadata["abstract"]
    if metadata.get("doi"):
        fields["DOI"] = metadata["doi"]

    item = ZoteroItemCreate(
        itemType=zotero_item_type,
        fields=fields,
        creators=creators,
        tags=["thala-research", "auto-citation"],
    )

    try:
        return await store_manager.zotero.add(item)
    except Exception as e:
        logger.error(f"Failed to create Zotero item for {url}: {e}")
        return None
