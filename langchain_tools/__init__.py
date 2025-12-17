"""
LangChain tools for Thala memory system.

Tools provided:
- search_memory: Cross-store semantic search
- expand_context: Deep-dive retrieval ("more about that")
- web_search: Search the web for information
- scrape_url: Scrape a webpage to markdown
- map_website: Discover URLs on a website
- process_document: Document extraction and summarization pipeline
"""

from .base import StoreManager, get_store_manager
from .search_memory import (
    search_memory,
    MemorySearchResult,
    SearchMemoryOutput,
)
from .expand_context import (
    expand_context,
    ExpandedContext,
)
from .firecrawl import (
    web_search,
    scrape_url,
    map_website,
    WebSearchResult,
    WebSearchOutput,
    ScrapeOutput,
    MapOutput,
)
from .document_processing import (
    process_document,
    DocumentProcessingOutput,
)

__all__ = [
    # Store management
    "StoreManager",
    "get_store_manager",
    # search_memory tool
    "search_memory",
    "MemorySearchResult",
    "SearchMemoryOutput",
    # expand_context tool
    "expand_context",
    "ExpandedContext",
    # firecrawl tools
    "web_search",
    "scrape_url",
    "map_website",
    "WebSearchResult",
    "WebSearchOutput",
    "ScrapeOutput",
    "MapOutput",
    # document_processing tool
    "process_document",
    "DocumentProcessingOutput",
]
