# LangChain Tools

LangChain 1.x tools for integrating Thala stores, web search, and research workflows.

## Available Tools

### Memory & Store Search

| Tool | Purpose |
|------|---------|
| `search_memory` | Cross-store semantic search (top_of_mind, coherence, store, optionally who_i_was) |
| `expand_context` | Deep-dive retrieval for follow-up questions |
| `search_store` | Main store with language/type filters |
| `search_coherence` | Beliefs/preferences with confidence filters |
| `search_top_of_mind` | Active projects semantic search |
| `search_history` | Historical versions (who_i_was) |
| `search_forgotten` | Archived content |

### Web Search

| Tool | Purpose |
|------|---------|
| `web_search` | Web search via Firecrawl |
| `scrape_url` | URL to markdown conversion |
| `map_website` | URL discovery on websites |
| `perplexity_search` | AI-powered web search |
| `check_fact` | Fact verification |

### Research

| Tool | Purpose |
|------|---------|
| `openalex_search` | Academic literature (240M+ works) |
| `book_search` | Book/textbook discovery |
| `process_document` | Document extraction pipeline |
| `search_papers` | Hybrid search for papers in the corpus |
| `get_paper_content` | Fetch detailed paper content by Zotero key |

## Usage

```python
from langchain.agents import create_agent
from langchain_tools import (
    search_memory,
    expand_context,
    openalex_search,
)

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[search_memory, expand_context, openalex_search],
)
```

## Store Manager

Singleton providing lazy-initialized access to all stores:

```python
from langchain_tools import get_store_manager

manager = get_store_manager()
# Stores initialized on first access
await manager.elasticsearch.store.search("query")
await manager.chroma.search("query")
await manager.cleanup()  # Proper async cleanup
```

## OpenAlex Integration (`openalex/`)

Academic literature search and citation tracking:

```python
from langchain_tools.openalex import (
    openalex_search,
    get_forward_citations,
    get_backward_citations,
    get_author_works,
    get_work_by_doi,
)

# Search tool (for LangChain agents)
results = await openalex_search.ainvoke({"query": "machine learning", "limit": 10})

# Direct query functions (for programmatic use)
citations = await get_forward_citations(doi="10.1234/example")
author_works = await get_author_works(author_id="A1234567890")
```

## Output Types

Each tool returns a typed output:

- `SearchMemoryOutput` - Cross-store results
- `ExpandedContext` - Deep retrieval results
- `StoreSearchOutput` - Store-specific search results
- `WebSearchOutput` - Web search results
- `ScrapeOutput` - Scraped webpage content
- `MapOutput` - Website URL discovery results
- `PerplexitySearchOutput` - Perplexity search results
- `FactCheckOutput` - Fact verification results
- `OpenAlexSearchOutput` - Academic results
- `BookSearchOutput` - Book results
- `DocumentProcessingOutput` - Document processing results
- `PaperSearchOutput` - Paper corpus search results
- `PaperContentOutput` - Paper content retrieval results

## Environment Variables

- `FIRECRAWL_API_KEY` - Firecrawl API key
- `PERPLEXITY_API_KEY` - Perplexity API key
- Store configuration via `THALA_*` variables (see core/README.md)
