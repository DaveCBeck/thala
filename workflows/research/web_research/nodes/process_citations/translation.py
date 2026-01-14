"""Translation server integration."""

import logging
from typing import Optional

from core.stores.translation_server import TranslationServerClient, TranslationResult
from core.stores.zotero import ZoteroSearchCondition

logger = logging.getLogger(__name__)


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


async def _check_existing_zotero_item(
    url: str,
    store_manager,
) -> Optional[str]:
    """Check if a Zotero item already exists for this URL."""
    try:
        results = await store_manager.zotero.search(
            [ZoteroSearchCondition(condition="url", operator="is", value=url)], limit=1
        )
        if results:
            return results[0].key
    except Exception as e:
        logger.debug(f"Error checking existing Zotero item: {e}")
    return None
