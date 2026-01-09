"""Core citation processing logic."""

import asyncio
import logging
from typing import Any

from core.stores.translation_server import TranslationServerClient
from langchain_tools.base import get_store_manager
from workflows.web_research.state import DeepResearchState

from .citation_formatter import _replace_citations_in_report
from .metadata import _enhance_metadata_with_llm
from .translation import _check_existing_zotero_item, _get_translation_metadata
from .utils import _normalize_url
from .zotero_creator import _create_zotero_item

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 3


async def _process_single_citation(
    citation: dict,
    translation_client: TranslationServerClient,
    store_manager,
    scraped_content_map: dict,
    source_metadata_map: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """
    Process a single citation.

    Returns:
        Tuple of (url, zotero_key) where zotero_key may be None on failure
    """
    url = citation.get("url", "")
    if not url:
        return (url, None)

    async with semaphore:
        try:
            # Check for existing Zotero item first
            existing_key = await _check_existing_zotero_item(url, store_manager)
            if existing_key:
                logger.info(f"Found existing Zotero item {existing_key} for: {url[:50]}...")
                return (url, existing_key)

            # Check if we have OpenAlex metadata for this URL
            source_metadata = source_metadata_map.get(url)
            if source_metadata and source_metadata.get("source_type") == "openalex":
                # Use OpenAlex metadata directly - skip Translation Server
                logger.info(f"Using OpenAlex metadata for: {url[:50]}...")

                enhanced_metadata = {
                    "title": citation.get("title") or source_metadata.get("title"),
                    "authors": [
                        a.get("name") for a in source_metadata.get("authors", [])
                        if a.get("name")
                    ],
                    "date": source_metadata.get("publication_date"),
                    "publication_title": source_metadata.get("source_name"),
                    "abstract": source_metadata.get("abstract"),
                    "doi": source_metadata.get("doi"),
                    "item_type": "journalArticle",  # Academic source
                }

                # Use DOI for Zotero URL if available
                zotero_url = source_metadata.get("doi") or url
                zotero_key = await _create_zotero_item(zotero_url, enhanced_metadata, store_manager)

                if zotero_key:
                    logger.info(f"Created Zotero item {zotero_key} (OpenAlex) for: {url[:50]}...")
                else:
                    logger.warning(f"Failed to create Zotero item for: {url[:50]}...")

                return (url, zotero_key)

            # Standard flow: Translation Server + LLM enhancement
            # Small delay to be polite to translation server
            await asyncio.sleep(0.3)

            # Step 1: Get translation metadata
            translation_result = await _get_translation_metadata(url, translation_client)

            # Step 2: Enhance with LLM (passing translation result including empty fields)
            scraped_content = scraped_content_map.get(url)
            enhanced_metadata = await _enhance_metadata_with_llm(
                translation_result, url, scraped_content
            )

            # Step 3: Create Zotero item
            zotero_key = await _create_zotero_item(url, enhanced_metadata, store_manager)

            if zotero_key:
                logger.info(f"Created Zotero item {zotero_key} for: {url[:50]}...")
            else:
                logger.warning(f"Failed to create Zotero item for: {url[:50]}...")

            return (url, zotero_key)

        except Exception as e:
            logger.error(f"Failed to process citation {url}: {e}")
            return (url, None)


async def process_citations(state: DeepResearchState) -> dict[str, Any]:
    """
    Process citations and convert to Pandoc-style cite keys.

    This node:
    1. Gets metadata for each URL via Translation Server
    2. Enhances metadata with LLM using any scraped content
    3. Creates Zotero items with full metadata
    4. Replaces [1], [2] citations with [@ZOTEROKEY]

    Returns:
        - final_report: Updated report with Pandoc citations
        - citations: Updated citations with zotero_keys
        - citation_keys: List of created Zotero keys
        - current_status: Updated status
    """
    report = state.get("final_report")
    citations = state.get("citations", [])
    findings = state.get("research_findings", [])

    if not report or not citations:
        logger.info("No report or citations to process")
        return {"current_status": "saving_findings"}

    logger.info(f"Processing {len(citations)} citations")

    store_manager = get_store_manager()

    # Build scraped content map and source metadata map from findings
    scraped_content_map = {}
    source_metadata_map = {}
    for finding in findings:
        for source in finding.get("sources", []):
            url = source.get("url", "")
            content = source.get("content")
            source_metadata = source.get("source_metadata")

            if url:
                if content:
                    scraped_content_map[url] = content
                if source_metadata:
                    source_metadata_map[url] = source_metadata

                # Also map normalized URL
                normalized = _normalize_url(url)
                if normalized != url:
                    if content:
                        scraped_content_map[normalized] = content
                    if source_metadata:
                        source_metadata_map[normalized] = source_metadata

    logger.debug(
        f"Built maps: {len(scraped_content_map)} scraped, "
        f"{len(source_metadata_map)} with source_metadata"
    )

    # Process all citations concurrently (with limit)
    translation_client = TranslationServerClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    try:
        tasks = [
            _process_single_citation(
                citation,
                translation_client,
                store_manager,
                scraped_content_map,
                source_metadata_map,
                semaphore,
            )
            for citation in citations
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build URL -> key mapping
        url_to_key = {}
        citation_keys = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Citation processing error: {result}")
                continue
            url, key = result
            if key:
                url_to_key[url] = key
                citation_keys.append(key)

        # Update citations with keys
        updated_citations = []
        for citation in citations:
            url = citation.get("url", "")
            updated = dict(citation)
            if url in url_to_key:
                updated["zotero_key"] = url_to_key[url]
            updated_citations.append(updated)

        # Replace citations in report
        updated_report = _replace_citations_in_report(
            report, url_to_key, citations
        )

        logger.info(
            f"Processed {len(citations)} citations, "
            f"created/linked {len(url_to_key)} Zotero items"
        )

        return {
            "final_report": updated_report,
            "citations": updated_citations,
            "citation_keys": citation_keys,
            "current_status": "saving_findings",
        }

    finally:
        await translation_client.close()
