"""LLM-based metadata enhancement."""

import json
import logging
from typing import Optional

from core.stores.translation_server import TranslationResult
from workflows.shared.llm_utils import ModelTier, extract_json

logger = logging.getLogger(__name__)

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
