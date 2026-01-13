# LangChain Tools

LangChain 1.x tools for integrating Thala stores, web search, and research workflows.

## Available Tools

### Memory & Store Search

| Tool | Purpose |
|------|---------|
| `search_memory` | Cross-store semantic search (all 5 stores) |
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

Async client for academic literature:

```python
from langchain_tools.openalex import OpenAlexClient

client = OpenAlexClient()
works = await client.search_works("machine learning", per_page=25)
citations = await client.get_citations(work_id)
```

## Output Types

Each tool returns a typed output:

- `SearchMemoryOutput` - Cross-store results
- `ExpandedContext` - Deep retrieval results
- `WebSearchOutput` - Web search results
- `OpenAlexSearchOutput` - Academic results
- `BookSearchOutput` - Book results

## Environment Variables

- `FIRECRAWL_API_KEY` - Firecrawl API key
- `PERPLEXITY_API_KEY` - Perplexity API key
- Store configuration via `THALA_*` variables (see core/README.md)
