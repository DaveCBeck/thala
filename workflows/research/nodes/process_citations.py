"""
Process citations node.

Post-processes the final report to:
1. Extract metadata for each cited URL via Translation Server
2. Enhance metadata with LLM using scraped content
3. Create Zotero items for each citation
4. Replace numeric citations [1], [2] with Pandoc-style [@KEY]
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional
from urllib.parse import urlparse

from core.stores.translation_server import TranslationServerClient, TranslationResult
from core.stores.zotero import ZoteroItemCreate, ZoteroCreator, ZoteroSearchCondition
from langchain_tools.base import get_store_manager
from workflows.research.state import DeepResearchState
from workflows.shared.llm_utils import ModelTier, extract_json

logger = logging.getLogger(__name__)

# Maximum concurrent citation processing
MAX_CONCURRENT = 3

# Prompt for LLM metadata enhancement
METADATA_ENHANCEMENT_PROMPT = """You are extracting/improving bibliographic metadata for a web source.

The Zotero Translation Server provided this metadata (fields may be empty or missing):
<translation_result>
{translation_json}
</translation_result>

Content from the source page:
<page_content>
{page_content}
</page_content>

Your task:
1. Fill in any EMPTY or null fields using information from the page content
2. Correct any obviously wrong metadata
3. Add publication date if you can determine it
4. Extract author names in proper format (First Last)

Return ONLY a JSON object with these fields:
- title: string (required - use the page content to determine if empty)
- authors: list of author names as strings (e.g., ["John Smith", "Jane Doe"])
- date: publication date as string (YYYY or YYYY-MM-DD format), null if unknown
- publication_title: name of the publication/website/journal
- abstract: brief description if available (1-2 sentences max)
- doi: DOI if mentioned in the content
- item_type: Zotero item type (webpage, journalArticle, blogPost, report, newspaperArticle, magazineArticle)

Use null for fields you cannot determine. Be accurate - don't guess."""

METADATA_SCHEMA = """{
  "title": "string",
  "authors": ["string"],
  "date": "string or null",
  "publication_title": "string or null",
  "abstract": "string or null",
  "doi": "string or null",
  "item_type": "string"
}"""


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    parsed = urlparse(url)
    # Remove trailing slashes, normalize to lowercase domain
    path = parsed.path.rstrip("/") if parsed.path != "/" else ""
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"


async def _get_translation_metadata(
    url: str,
    translation_client: TranslationServerClient,
) -> Optional[TranslationResult]:
    """Get metadata from translation server."""
    try:
        return await translation_client.translate_url(url)
    except Exception as e:
        logger.warning(f"Translation failed for {url}: {e}")
        return None


async def _enhance_metadata_with_llm(
    translation_result: Optional[TranslationResult],
    url: str,
    scraped_content: Optional[str],
) -> dict:
    """
    Enhance translation result with LLM analysis.

    The LLM fills in gaps and improves metadata using the scraped content.
    Passes translation result (including empty fields) so LLM knows what to fill.
    """
    # Build translation JSON for prompt
    if translation_result:
        translation_dict = translation_result.to_dict_for_llm()
    else:
        translation_dict = {
            "itemType": "webpage",
            "url": url,
            "title": None,
            "authors": [],
            "date": None,
            "abstractNote": None,
            "publicationTitle": None,
            "DOI": None,
            "ISSN": None,
            "ISBN": None,
            "language": None,
            "publisher": None,
            "volume": None,
            "issue": None,
            "pages": None,
        }

    translation_json = json.dumps(translation_dict, indent=2)

    # Truncate content to avoid token limits (first 6000 chars)
    content = scraped_content[:6000] if scraped_content else "No content available."

    try:
        enhanced = await extract_json(
            text=content,
            prompt=METADATA_ENHANCEMENT_PROMPT.format(
                translation_json=translation_json,
                page_content=content,
            ),
            schema_hint=METADATA_SCHEMA,
            tier=ModelTier.HAIKU,  # Use Haiku for cost efficiency
        )
        return enhanced
    except Exception as e:
        logger.warning(f"LLM metadata enhancement failed: {e}")
        # Return basic metadata from translation result
        if translation_result:
            return {
                "title": translation_result.title or url,
                "authors": [c.to_full_name() for c in translation_result.creators],
                "date": translation_result.date,
                "publication_title": translation_result.publication_title or translation_result.website_title,
                "abstract": translation_result.abstract_note,
                "doi": translation_result.doi,
                "item_type": translation_result.item_type,
            }
        return {"title": url, "item_type": "webpage", "authors": []}


async def _check_existing_zotero_item(
    url: str,
    store_manager,
) -> Optional[str]:
    """Check if a Zotero item already exists for this URL."""
    try:
        results = await store_manager.zotero.search([
            ZoteroSearchCondition(condition="url", operator="is", value=url)
        ], limit=1)
        if results:
            return results[0].key
    except Exception as e:
        logger.debug(f"Error checking existing Zotero item: {e}")
    return None


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
                creators.append(ZoteroCreator(
                    firstName=parts[0],
                    lastName=parts[1],
                    creatorType="author",
                ))
            else:
                creators.append(ZoteroCreator(
                    name=author,
                    creatorType="author",
                ))

    # Map item types to Zotero types
    item_type = metadata.get("item_type", "webpage")
    type_mapping = {
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
    zotero_item_type = type_mapping.get(item_type, "webpage")

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


async def _process_single_citation(
    citation: dict,
    translation_client: TranslationServerClient,
    store_manager,
    scraped_content_map: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Optional[str]]:
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


def _replace_citations_in_report(
    report: str,
    url_to_key: dict[str, str],
    citations: list[dict],
) -> str:
    """
    Replace numeric citations with Pandoc-style cite keys.

    Transforms:
    - Inline: [1] -> [@ABCD1234]
    - References section: [1] Title: URL -> [@ABCD1234] Title
    """
    # Build index-to-key mapping (same order as _extract_citations)
    index_to_key = {}
    seen_urls = set()
    idx = 1

    for citation in citations:
        url = citation.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            if url in url_to_key and url_to_key[url]:
                index_to_key[idx] = url_to_key[url]
            idx += 1

    if not index_to_key:
        return report

    # Replace inline citations [1], [2], etc. with [@KEY]
    def replace_inline(match):
        idx = int(match.group(1))
        if idx in index_to_key:
            return f"[@{index_to_key[idx]}]"
        return match.group(0)  # Keep original if no mapping

    updated_report = re.sub(r'\[(\d+)\]', replace_inline, report)

    # Update references section
    # Pattern: [N] Title: URL -> [@KEY] Title
    def replace_reference(match):
        idx = int(match.group(1))
        title = match.group(2)
        if idx in index_to_key:
            return f"[@{index_to_key[idx]}] {title}"
        return match.group(0)

    updated_report = re.sub(
        r'^\[(\d+)\]\s+(.+?):\s+https?://\S+\s*$',
        replace_reference,
        updated_report,
        flags=re.MULTILINE,
    )

    return updated_report


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

    # Build scraped content map from findings
    scraped_content_map = {}
    for finding in findings:
        for source in finding.get("sources", []):
            url = source.get("url", "")
            content = source.get("content")
            if url and content:
                scraped_content_map[url] = content
                # Also map normalized URL
                normalized = _normalize_url(url)
                if normalized != url:
                    scraped_content_map[normalized] = content

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
