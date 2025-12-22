"""
LangChain tool for searching books.

Searches for books by title, author, ISBN, or topic and returns
structured results with metadata.

Requires the retrieve-academic service to be running. Falls back gracefully
if the service is unavailable.
"""

import logging
import os
from enum import Enum
from typing import Optional

from cachetools import TTLCache
from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy singleton client for retrieve-academic service
_book_search_client = None

# Default service URL (can be overridden by env var)
DEFAULT_SERVICE_URL = "http://localhost:8002"

# Cache for search results (30 min TTL, max 100 items)
_search_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)


class BookFormat(str, Enum):
    """Supported book formats."""

    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"
    MOBI = "mobi"
    AZW3 = "azw3"
    FB2 = "fb2"
    DJVU = "djvu"
    CBZ = "cbz"
    CBR = "cbr"
    OTHER = "other"


# Format priority for sorting results (lower = better)
FORMAT_PRIORITY: dict[str, int] = {
    BookFormat.PDF: 1,
    BookFormat.EPUB: 2,
    BookFormat.TXT: 3,
    BookFormat.MOBI: 4,
    BookFormat.AZW3: 5,
    BookFormat.FB2: 6,
    BookFormat.DJVU: 7,
    BookFormat.CBZ: 8,
    BookFormat.CBR: 9,
    BookFormat.OTHER: 10,
}


class Book(BaseModel):
    """Individual book from search results."""

    title: str
    authors: str
    publisher: Optional[str] = None
    language: str = "Unknown"
    format: str = "other"
    size: str = "Unknown"
    md5: str
    url: str
    abstract: Optional[str] = None


class BookSearchOutput(BaseModel):
    """Output schema for book_search tool."""

    query: str
    total_results: int
    results: list[Book] = Field(default_factory=list)


def _get_client():
    """Get httpx AsyncClient for retrieve-academic service (lazy init)."""
    global _book_search_client
    if _book_search_client is None:
        import httpx

        service_url = os.environ.get("BOOK_SEARCH_SERVICE_URL", DEFAULT_SERVICE_URL)
        _book_search_client = httpx.AsyncClient(
            base_url=service_url,
            timeout=30.0,
        )
    return _book_search_client


async def _search_books_internal(query: str, limit: int = 10) -> BookSearchOutput:
    """
    Search for books via the retrieve-academic service.

    Falls back gracefully if the service is unavailable.
    """
    cache_key = f"search:{query}:{limit}"

    # Check cache
    if cache_key in _search_cache:
        logger.debug(f"Cache hit for book search: {query}")
        return _search_cache[cache_key]

    try:
        client = _get_client()
        response = await client.post(
            "/search",
            json={"query": query, "limit": limit},
        )
        response.raise_for_status()
        data = response.json()

        # Convert service response to Book objects
        books: list[Book] = []
        for r in data.get("results", []):
            books.append(
                Book(
                    title=r.get("title", ""),
                    authors=r.get("authors", "Unknown"),
                    publisher=r.get("publisher"),
                    language=r.get("language", "Unknown"),
                    format=r.get("format", "other"),
                    size=r.get("size", "Unknown"),
                    md5=r.get("md5", ""),
                    url=r.get("url", ""),
                    abstract=r.get("abstract"),
                )
            )

        output = BookSearchOutput(
            query=query,
            total_results=len(books),
            results=books,
        )

        # Cache the result
        _search_cache[cache_key] = output

        logger.debug(f"book_search returned {len(books)} results for '{query}'")
        return output

    except Exception as e:
        # Log at debug level - service might not be running
        logger.debug(f"Book search service unavailable: {e}")
        return BookSearchOutput(query=query, total_results=0, results=[])


@tool
async def book_search(query: str, limit: int = 10) -> dict:
    """Search for books by title, author, ISBN, or topic.

    Use this to find books, textbooks, and other long-form written content.
    Results include format (PDF, EPUB, etc.), size, and download identifiers.

    Args:
        query: Search query - can be title, author name, ISBN, or topic
        limit: Maximum number of results to return (default: 10, max: 50)

    Returns:
        Book search results with title, authors, format, size, and metadata.
    """
    limit = min(max(1, limit), 50)

    try:
        output = await _search_books_internal(query, limit)
        return output.model_dump(mode="json")
    except Exception as e:
        logger.error(f"book_search failed: {e}")
        return BookSearchOutput(
            query=query,
            total_results=0,
            results=[],
        ).model_dump(mode="json")
