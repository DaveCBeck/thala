"""
Async client for Zotero Translation Server.

The Translation Server extracts bibliographic metadata from URLs using
Zotero's translator framework. This provides structured citation data
without needing the full Zotero client.

Endpoints:
- POST /web: Extract metadata from a webpage URL
- POST /search: Lookup by identifier (DOI, ISBN, PMID, arXiv ID)
- POST /export: Convert Zotero JSON to export format (BibTeX, RIS, etc.)

Reference: https://github.com/zotero/translation-server
"""

import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, ConfigDict

from core.utils import BaseAsyncHttpClient, generate_cache_key
from workflows.shared.persistent_cache import get_cached, set_cached

logger = logging.getLogger(__name__)


class TranslationCreator(BaseModel):
    """Creator (author) from translation result."""

    model_config = ConfigDict(populate_by_name=True)

    first_name: Optional[str] = Field(None, alias="firstName")
    last_name: Optional[str] = Field(None, alias="lastName")
    name: Optional[str] = None  # Single-field name (organizations, etc.)
    creator_type: str = Field("author", alias="creatorType")

    def to_full_name(self) -> str:
        """Convert to full name string."""
        if self.name:
            return self.name
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) if parts else "Unknown"


class TranslationResult(BaseModel):
    """Result from translation server."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    item_type: str = Field("webpage", alias="itemType")
    title: Optional[str] = None
    creators: list[TranslationCreator] = Field(default_factory=list)
    date: Optional[str] = None
    url: Optional[str] = None
    abstract_note: Optional[str] = Field(None, alias="abstractNote")
    publication_title: Optional[str] = Field(None, alias="publicationTitle")
    website_title: Optional[str] = Field(None, alias="websiteTitle")
    doi: Optional[str] = Field(None, alias="DOI")
    issn: Optional[str] = Field(None, alias="ISSN")
    isbn: Optional[str] = Field(None, alias="ISBN")
    language: Optional[str] = None
    access_date: Optional[str] = Field(None, alias="accessDate")
    publisher: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None

    def to_dict_for_llm(self) -> dict[str, Any]:
        """Convert to dict format suitable for LLM enhancement.

        Includes all fields, even empty ones, so LLM knows what to fill.
        """
        return {
            "itemType": self.item_type,
            "title": self.title,
            # pylint: disable=not-an-iterable  # Pydantic Field(default_factory=list)
            "authors": [c.to_full_name() for c in self.creators]
            if self.creators
            else [],
            "date": self.date,
            "url": self.url,
            "abstractNote": self.abstract_note,
            "publicationTitle": self.publication_title or self.website_title,
            "DOI": self.doi,
            "ISSN": self.issn,
            "ISBN": self.isbn,
            "language": self.language,
            "publisher": self.publisher,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
        }


class TranslationServerClient(BaseAsyncHttpClient):
    """
    Async client for Zotero Translation Server.

    Example:
        async with TranslationServerClient() as client:
            result = await client.translate_url("https://example.com/article")
            if result:
                print(f"Title: {result.title}")
                print(f"Authors: {[c.to_full_name() for c in result.creators]}")
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
    ):
        super().__init__(
            host=host,
            port=port,
            timeout=timeout,
            host_env_var="THALA_TRANSLATION_HOST",
            port_env_var="THALA_TRANSLATION_PORT",
            host_default="localhost",
            port_default=1969,
        )

    async def translate_url(self, url: str) -> Optional[TranslationResult]:
        """
        Translate a URL to bibliographic metadata.

        Args:
            url: Web page URL to translate

        Returns:
            TranslationResult with extracted metadata, or None if translation failed
        """
        cache_key = generate_cache_key(url)
        cached_result = get_cached("translation_server", cache_key, ttl_days=30)
        if cached_result is not None:
            return cached_result

        client = await self._get_client()

        try:
            # Use JSON format (required by zuphilip/translation-server image)
            response = await client.post(
                "/web",
                json={"url": url, "sessionid": "thala"},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                # Translation server returns array of items
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    # Parse creators if present
                    if "creators" in item:
                        item["creators"] = [
                            TranslationCreator.model_validate(c)
                            for c in item.get("creators", [])
                        ]
                    result = TranslationResult.model_validate(item)
                    set_cached("translation_server", cache_key, result)
                    return result

            elif response.status_code == 300:
                # Multiple results - server returns selection dialog info
                # We'll create a basic result with the URL
                logger.debug(
                    f"Multiple translation matches for {url}, using basic result"
                )
                result = TranslationResult(
                    item_type="webpage",
                    url=url,
                )
                set_cached("translation_server", cache_key, result)
                return result

            elif response.status_code == 501:
                # No translator available for this URL
                logger.debug(f"No translator available for {url}")
                return None

            elif response.status_code == 400:
                # Bad request - possibly content-type issue
                logger.debug(f"Bad request for {url}: {response.text}")
                return None

            else:
                logger.warning(
                    f"Translation failed for {url}: HTTP {response.status_code}"
                )

            return None

        except httpx.ConnectError:
            logger.warning(f"Translation server unavailable at {self.base_url}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Translation timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Translation server error for {url}: {e}")
            return None

    async def search_identifier(self, identifier: str) -> Optional[TranslationResult]:
        """
        Lookup metadata by identifier (DOI, ISBN, PMID, arXiv ID).

        Args:
            identifier: DOI, ISBN, PMID, or arXiv ID

        Returns:
            TranslationResult or None if not found
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/search",
                content=identifier,
                headers={"Content-Type": "text/plain"},
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    if "creators" in item:
                        item["creators"] = [
                            TranslationCreator.model_validate(c)
                            for c in item.get("creators", [])
                        ]
                    return TranslationResult.model_validate(item)

            return None

        except Exception as e:
            logger.error(f"Identifier search failed for {identifier}: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if translation server is available."""
        try:
            client = await self._get_client()
            response = await client.get("/", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
