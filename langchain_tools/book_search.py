"""
LangChain tool for searching books.

Searches for books by title, author, ISBN, or topic and returns
structured results with metadata.
"""

import logging
import os
import re
from enum import Enum
from typing import Optional

from bs4 import BeautifulSoup
from cachetools import TTLCache
from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy singleton client
_book_search_client = None

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
    """Get httpx AsyncClient for book search (lazy init)."""
    global _book_search_client
    if _book_search_client is None:
        import httpx

        base_url = os.environ.get("BOOK_SEARCH_BASE_URL")
        if not base_url:
            raise ValueError("BOOK_SEARCH_BASE_URL environment variable is required")
        _book_search_client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            follow_redirects=True,
        )
    return _book_search_client


def _parse_format(format_str: str) -> str:
    """Parse format string from metadata."""
    if not format_str:
        return BookFormat.OTHER

    cleaned = format_str.replace("[", "").replace("]", "").lower().strip()

    if "pdf" in cleaned:
        return BookFormat.PDF
    if "epub" in cleaned:
        return BookFormat.EPUB
    if "txt" in cleaned or "text" in cleaned:
        return BookFormat.TXT
    if "mobi" in cleaned:
        return BookFormat.MOBI
    if "azw3" in cleaned:
        return BookFormat.AZW3
    if "fb2" in cleaned:
        return BookFormat.FB2
    if "djvu" in cleaned:
        return BookFormat.DJVU
    if "cbz" in cleaned:
        return BookFormat.CBZ
    if "cbr" in cleaned:
        return BookFormat.CBR

    return BookFormat.OTHER


def _truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to maximum length."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].strip() + "..."


async def _search_books_internal(query: str, limit: int = 10) -> BookSearchOutput:
    """
    Search for books.

    Parses HTML results and returns structured book data.
    """
    cache_key = f"search:{query}:{limit}"

    # Check cache
    if cache_key in _search_cache:
        logger.debug(f"Cache hit for book search: {query}")
        return _search_cache[cache_key]

    client = _get_client()
    base_url = os.environ.get("BOOK_SEARCH_BASE_URL", "")

    try:
        response = await client.get("/search", params={"q": query, "page": 1})
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Book search request failed: {e}")
        return BookSearchOutput(query=query, total_results=0, results=[])

    # Parse HTML response
    soup = BeautifulSoup(response.text, "lxml")
    books: list[Book] = []

    # Find all book containers
    containers = soup.select("div.flex.pt-3.pb-3.border-b")
    logger.debug(f"Found {len(containers)} book containers")

    for container in containers:
        # Find MD5 link
        md5_link = container.select_one('a[href^="/md5/"]')
        if not md5_link:
            continue

        href = md5_link.get("href", "")
        md5 = href.replace("/md5/", "")

        # Find title
        title_elem = container.select_one(
            r"a.line-clamp-\[3\].overflow-hidden.break-words"
        )
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Find authors
        author_elem = container.select_one(r"a:has(.icon-\[mdi--user-edit\])")
        authors = author_elem.get_text(strip=True) if author_elem else "Unknown"

        # Find publisher
        publisher_elem = container.select_one(r"a:has(.icon-\[mdi--company\])")
        publisher = (
            publisher_elem.get_text(strip=True) if publisher_elem else None
        )

        # Find metadata line
        meta_elem = container.select_one(
            r"div.text-gray-800.dark\:text-slate-400"
        )
        meta_text = meta_elem.get_text(strip=True) if meta_elem else ""

        # Parse metadata: "... Language [code] ... FORMAT ... SIZE ... YEAR ..."
        language = "Unknown"
        book_format = BookFormat.OTHER
        size = "Unknown"

        meta_match = re.search(
            r"✅\s*([^[]+)\[([^\]]+)\]\s*·\s*([^·]+)\s*·\s*([^·]+)\s*·\s*(\d{4})",
            meta_text,
        )
        if meta_match:
            language = meta_match.group(1).strip()
            book_format = _parse_format(meta_match.group(3).strip())
            size = meta_match.group(4).strip()

        # Find abstract/description
        desc_elem = container.select_one(
            r"div.line-clamp-\[2\].overflow-hidden.break-words.text-sm.text-gray-600"
        )
        abstract = desc_elem.get_text(strip=True) if desc_elem else None
        if abstract:
            abstract = _truncate_text(abstract, 200)

        if title and md5:
            books.append(
                Book(
                    title=title,
                    authors=authors,
                    publisher=publisher,
                    language=language,
                    format=book_format,
                    size=size,
                    md5=md5,
                    url=f"{base_url}/md5/{md5}",
                    abstract=abstract,
                )
            )

    # Sort by format priority
    books.sort(key=lambda b: FORMAT_PRIORITY.get(b.format, 10))

    # Limit results
    books = books[:limit]

    output = BookSearchOutput(
        query=query,
        total_results=len(books),
        results=books,
    )

    # Cache the result
    _search_cache[cache_key] = output

    logger.debug(f"book_search returned {len(books)} results for '{query}'")
    return output


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
