"""
LangChain tools for Thala memory system.

Tools provided:
- search_memory: Cross-store semantic search
- expand_context: Deep-dive retrieval ("more about that")
- web_search: Search the web for information (Firecrawl)
- scrape_url: Scrape a webpage to markdown
- map_website: Discover URLs on a website
- perplexity_search: AI-powered web search (Perplexity)
- check_fact: Fact verification using Perplexity
- openalex_search: Academic literature search (OpenAlex)
- book_search: Book search for books, textbooks, and publications
- process_document: Document extraction and summarization pipeline
- deep_research: Comprehensive research with memory integration
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
from .deep_research import (
    deep_research,
    DeepResearchOutput,
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
    # deep_research tool
    "deep_research",
    "DeepResearchOutput",
]
