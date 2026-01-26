# Core

Foundation layer providing shared infrastructure for all workflows and tools.

## Components

```
core/
├── config.py              # Environment configuration, logging, LangSmith tracing
├── embedding.py           # Embedding service (Voyage/OpenAI/Ollama)
├── scraping/              # Unified URL content retrieval
│   ├── unified.py         # Primary interface: get_url()
│   ├── service.py         # Lower-level scraper with fallback chain
│   ├── doi/               # DOI detection and resolution
│   ├── pdf/               # PDF download and processing via Marker
│   ├── classification/    # Academic content detection
│   ├── fallback/          # retrieve-academic integration
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
├── images/                # Public domain image aggregator
│   └── service.py         # Pexels + Unsplash with LLM selection
├── task_queue/            # Workflow task management
│   ├── queue_manager.py   # Task scheduling and round-robin
│   ├── checkpoint_manager.py  # Checkpoint/resume capability
│   └── budget_tracker.py  # LangSmith cost aggregation
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

**Lazy initialization**: Services are created on first access, configured via environment variables.

```python
from core.stores import ZoteroStore, TranslationServerClient, RetrieveAcademicClient
from core.stores.chroma import ChromaStore
from core.stores.elasticsearch import ElasticsearchStores

# Zotero citation management
zotero = ZoteroStore()
items = await zotero.search_by_title("machine learning")

# Vector store (ChromaDB)
chroma = ChromaStore(host="localhost", port=8000)
results = await chroma.search(query_embedding, n_results=10)

# Elasticsearch stores
es = ElasticsearchStores()
await es.store.search({"match": {"content": "query"}})
await es.coherence.search({"match": {"content": "query"}})
```

**Environment configuration**:
- `THALA_ES_COHERENCE_HOST` - Coherence ES (default: `http://localhost:9201`)
- `THALA_ES_FORGOTTEN_HOST` - Forgotten ES (default: `http://localhost:9200`)
- `THALA_CHROMA_HOST/PORT` - Vector DB (default: `localhost:8000`)
- `THALA_MODE=dev` - Enable LangSmith tracing
- `THALA_LOG_LEVEL_CONSOLE` - Console log level (default: WARNING)
- `THALA_LOG_LEVEL_FILE` - File log level (default: INFO)
- `THALA_LOG_DIR` - Directory for log files (default: ./logs/)

## Embedding Service

Supports Voyage AI (default), OpenAI, and Ollama with automatic chunking and TTL caching (90-day default).

```python
from core.embedding import EmbeddingService

service = EmbeddingService()  # Uses THALA_EMBEDDING_PROVIDER (default: voyage)
embedding = await service.embed("single text")
embeddings = await service.embed_batch(["text1", "text2"])
long_embedding = await service.embed_long("very long text...")  # Auto-chunks and averages
```

**Environment variables**:
- `THALA_EMBEDDING_PROVIDER` - Provider: `voyage` (default), `openai`, or `ollama`
- `THALA_EMBEDDING_MODEL` - Model name (provider-specific defaults)
- `VOYAGE_API_KEY` - Required for Voyage AI
- `OPENAI_API_KEY` - Required for OpenAI
- `THALA_OLLAMA_HOST` - Ollama host (default: `http://localhost:11434`)

## Web Scraping

Unified URL content retrieval with DOI resolution, PDF processing, and intelligent fallback.

```python
from core.scraping import get_url, GetUrlOptions

# Simple usage - handles URLs, DOIs, PDFs automatically
result = await get_url("https://example.com")
result = await get_url("10.1038/nature12373")  # DOI
result = await get_url("https://arxiv.org/pdf/2301.00001.pdf")  # PDF

# With options
result = await get_url(url, GetUrlOptions(
    pdf_quality="quality",
    allow_retrieve_academic=True,
    detect_academic=True,
))

print(result.content)  # Markdown content
print(result.source)   # SCRAPED, PDF_DIRECT, RETRIEVE_ACADEMIC
print(result.doi)      # Detected or resolved DOI
```

**Fallback chain**:
1. DOI resolution via OpenAlex (if DOI detected)
2. PDF processing via Marker (if PDF URL)
3. Web scraping: Local Firecrawl -> Cloud Stealth -> Playwright
4. Content classification (academic detection)
5. retrieve-academic fallback (for paywalled content)

**Low-level scraping** (for direct control):
```python
from core.scraping import get_scraper_service

service = get_scraper_service()
result = await service.scrape("https://example.com")
print(result.markdown)
```
