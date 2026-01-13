"""LLM-based metadata enhancement."""

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from core.stores.translation_server import TranslationResult
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


class EnhancedMetadata(BaseModel):
    """Enhanced bibliographic metadata."""
    title: str = Field(description="Page/article title")
    authors: list[str] = Field(default_factory=list, description="Author names in 'First Last' format")
    date: Optional[str] = Field(default=None, description="Publication date (YYYY or YYYY-MM-DD)")
    publication_title: Optional[str] = Field(default=None, description="Publication/website/journal name")
    abstract: Optional[str] = Field(default=None, description="Brief description (1-2 sentences)")
    doi: Optional[str] = Field(default=None, description="DOI if mentioned")
    item_type: str = Field(default="webpage", description="Zotero item type")


METADATA_ENHANCEMENT_PROMPT = """You are extracting/improving bibliographic metadata for a web source.

The Zotero Translation Server provided this metadata (fields may be empty or missing):
<translation_result>
{translation_json}
</translation_result>

Your task:
1. Fill in any EMPTY or null fields using information from the page content
2. Correct any obviously wrong metadata
3. Add publication date if you can determine it
4. Extract author names in proper format (First Last)

Use null for fields you cannot determine. Be accurate - don't guess."""


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
        result = await get_structured_output(
            output_schema=EnhancedMetadata,
            user_prompt=f"Page content:\n{content}",
            system_prompt=METADATA_ENHANCEMENT_PROMPT.format(translation_json=translation_json),
            tier=ModelTier.HAIKU,
        )
        return result.model_dump()
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
