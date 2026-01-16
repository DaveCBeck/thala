"""Phase 4c: Fetch 10x book summaries from store."""

import logging
from typing import Any

from langchain_tools.base import get_store_manager

logger = logging.getLogger(__name__)


async def fetch_book_summaries(state: dict) -> dict[str, Any]:
    """Fetch 10x summaries for selected books from Elasticsearch store.

    Queries store_l2 (compression_level=2) by zotero_key to retrieve
    the detailed 10:1 summaries created during document processing.
    """
    selected_books = state.get("selected_books", [])

    if not selected_books:
        logger.info("No books selected, skipping summary fetch")
        return {
            "book_summaries_cache": {},
            "current_phase": "write_sections",
        }

    logger.info(f"Phase 4c: Fetching 10x summaries for {len(selected_books)} books")

    book_summaries_cache = {}

    try:
        store_manager = await get_store_manager()

        for book in selected_books:
            zotkey = book.get("zotero_key")
            if not zotkey:
                continue

            try:
                # Query store_l2 (compression_level=2) by zotero_key
                query = {
                    "bool": {
                        "must": [{"term": {"zotero_key.keyword": zotkey}}],
                        "filter": [{"term": {"compression_level": 2}}],
                    }
                }

                records = await store_manager.es_stores.store.search(
                    query=query,
                    compression_level=2,
                    size=1,
                )

                if records:
                    record = records[0]
                    # Get the summary text from the record
                    summary_text = record.text if hasattr(record, "text") else ""
                    if summary_text:
                        book_summaries_cache[zotkey] = summary_text
                        logger.debug(
                            f"Fetched 10x summary for [@{zotkey}]: {len(summary_text)} chars"
                        )
                    else:
                        logger.debug(f"No summary text in record for [@{zotkey}]")
                else:
                    logger.debug(f"No L2 record found for [@{zotkey}]")

            except Exception as e:
                logger.warning(f"Failed to fetch summary for [@{zotkey}]: {e}")

        logger.info(
            f"Fetched {len(book_summaries_cache)}/{len(selected_books)} book summaries"
        )

        return {
            "book_summaries_cache": book_summaries_cache,
            "current_phase": "write_sections",
        }

    except Exception as e:
        logger.error(f"Book summary fetch failed: {e}")
        return {
            "book_summaries_cache": {},
            "current_phase": "write_sections",
            "errors": [{"phase": "fetch_book_summaries", "error": str(e)}],
        }
