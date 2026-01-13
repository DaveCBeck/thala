"""
LangChain tools for Thala memory system.

Tools provided:
- search_memory: Cross-store semantic search
- expand_context: Deep-dive retrieval ("more about that")
- search_store: Main knowledge store (with language, type, compression filters)
- search_coherence: Beliefs/preferences (with category, confidence filters)
- search_top_of_mind: Active projects (semantic search with language filter)
- search_history: Historical versions
- search_forgotten: Archived content
- web_search: Search the web for information (Firecrawl)
- scrape_url: Scrape a webpage to markdown
- map_website: Discover URLs on a website
- perplexity_search: AI-powered web search (Perplexity)
- check_fact: Fact verification using Perplexity
- openalex_search: Academic literature search (OpenAlex)
- book_search: Book search for books, textbooks, and publications
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
from .store_search import (
    search_store,
    search_coherence,
    search_top_of_mind,
    search_history,
    search_forgotten,
    StoreSearchResult,
    StoreSearchOutput,
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
from .perplexity import (
    perplexity_search,
    check_fact,
    PerplexitySearchResult,
    PerplexitySearchOutput,
    FactCheckOutput,
)
from .openalex import (
    openalex_search,
    OpenAlexWork,
    OpenAlexAuthor,
    OpenAlexSearchOutput,
)
from .book_search import (
    book_search,
    Book,
    BookSearchOutput,
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
    # store_search tools
    "search_store",
    "search_coherence",
    "search_top_of_mind",
    "search_history",
    "search_forgotten",
    "StoreSearchResult",
    "StoreSearchOutput",
    # firecrawl tools
    "web_search",
    "scrape_url",
    "map_website",
    "WebSearchResult",
    "WebSearchOutput",
    "ScrapeOutput",
    "MapOutput",
    # perplexity tools
    "perplexity_search",
    "check_fact",
    "PerplexitySearchResult",
    "PerplexitySearchOutput",
    "FactCheckOutput",
    # openalex tools
    "openalex_search",
    "OpenAlexWork",
    "OpenAlexAuthor",
    "OpenAlexSearchOutput",
    # book_search tool
    "book_search",
    "Book",
    "BookSearchOutput",
    # document_processing tool
    "process_document",
    "DocumentProcessingOutput",
]
