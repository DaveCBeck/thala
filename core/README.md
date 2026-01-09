# Core

Foundation layer providing shared infrastructure for all workflows and tools.

## Components

```
core/
├── config.py              # Environment configuration, LangSmith tracing
├── embedding.py           # Embedding service (OpenAI/Ollama)
├── scraping/              # Multi-provider web scraping
│   ├── service.py         # Main scraper with Firecrawl + Playwright fallback
│   └── playwright_scraper.py
├── stores/                # Data persistence layer
│   ├── schema.py          # Unified record schema
│   ├── chroma.py          # Vector store (top_of_mind)
│   ├── elasticsearch/     # Structured storage
│   │   ├── client.py      # Async ES client
│   │   └── stores/        # Index-specific stores
│   ├── zotero/            # Citation management
│   ├── translation_server.py  # URL metadata extraction
│   └── retrieve_academic.py   # Academic document retrieval
└── utils/                 # Shared utilities
    ├── async_context.py   # Context managers
    ├── async_http_client.py
    └── caching.py         # TTL-based caching
```

## Stores

All stores use a unified schema with UUID primary keys, language codes, and embeddings. Every record references a `zotero_key` for source retrieval.

| Store | Backend | Purpose |
|-------|---------|---------|
| `top_of_mind` | Chroma | Active projects, fast vector retrieval |
| `coherence` | Elasticsearch | Identity, beliefs, preferences |
| `who_i_was` | Elasticsearch | Edit history (temporal queries) |
| `store` | Elasticsearch | Main knowledge, 10:1 compressions |
| `forgotten_store` | Elasticsearch | Archived with forgetting-reason |

## Key Patterns

**Lazy initialization**: Stores are created on first access, configured via environment variables.

```python
from core.stores import get_store_manager

manager = get_store_manager()
await manager.elasticsearch.store.search("query")
await manager.chroma.search("query")
```

**Environment configuration**:
- `THALA_ES_COHERENCE_HOST` - Coherence ES (default: `http://localhost:9201`)
- `THALA_ES_FORGOTTEN_HOST` - Forgotten ES (default: `http://localhost:9200`)
- `THALA_CHROMA_HOST/PORT` - Vector DB (default: `localhost:8000`)
- `THALA_MODE=dev` - Enable LangSmith tracing

## Embedding Service

Supports OpenAI and Ollama with automatic chunking and TTL caching (90-day default).

```python
from core.embedding import EmbeddingService

service = EmbeddingService()
embeddings = await service.embed_texts(["text1", "text2"])
```

## Web Scraping

Multi-provider scraping with automatic fallback: Firecrawl basic -> stealth -> Playwright.

```python
from core.scraping import scrape_url

result = await scrape_url("https://example.com")
print(result.markdown)
```
