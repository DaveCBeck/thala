# Thala

A personal knowledge system that goes beyond storage—it thinks alongside you.

For a decade I've run this as a done-for-you service: acting as a "second brain" for clients, knowing their thought processes and beliefs well enough to have a "mini-client" running in my subconscious. This project attempts to automate that relationship.

**The goal:** Give technical users who put the time in their own second brain—with background processes that mirror your own default mode network. It maintains coherence, recognizes patterns, and develops genuine context about *you* over time.

**Target audience:** Self-hosters comfortable with Docker. Particularly relevant if you use Obsidian, Roam Research, or other connected thinking tools and want something that actively processes rather than just stores.

**How it works:** The system maintains an LLM-generated "coherence" layer you never directly edit or view—this is the second brain's own understanding of who you are, what matters to you, and how things connect. Everything else flows from that.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Executive (conscious)                                           │
│   Directed research, writing, decisions—user-initiated tasks    │
├─────────────────────────────────────────────────────────────────┤
│ Subconscious (background)                                       │
│   Coherence maintenance, pattern recognition, goal alignment    │
│   Runs when idle—default mode network                           │
├─────────────────────────────────────────────────────────────────┤
│ Ingest Pipeline                                                 │
│   sources → marker → Zotero (full text) → stores (metadata)     │
├─────────────────────────────────────────────────────────────────┤
│ Stores (metadata + processed, NOT full text)                    │
│   top_of_mind │ coherence │ who_i_was │ store │ forgotten_store │
│   All reference Zotero item IDs for source retrieval            │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
thala/
├── docker-compose.yml        # Orchestrates all services
├── services/
│   ├── marker/
│   ├── zotero/
│   ├── elasticsearch/
│   └── chroma/
├── mcp_server/               # MCP server exposing store APIs
├── langchain_tools/          # LangChain tools (search_memory, expand_context)
├── mcps/
│   ├── zotero/
│   ├── firecrawl/
│   ├── search/
│   └── deep-research/
├── core/
│   ├── embedding.py          # Embedding service (OpenAI/Ollama)
│   ├── ingest/               # Sources → Zotero → stores pipeline
│   ├── executive/            # Conscious: user-initiated tasks
│   ├── subconscious/         # Background: coherence, patterns, goals
│   ├── stores/               # Abstractions over ES/Chroma + Zotero refs
│   └── outputs/              # Writing, contact stack
└── tests/
```

## Stores & Backends

**Zotero is source of truth for full text.** All stores below hold metadata, embeddings, processed derivatives, and connection graphs—never raw content. Every record includes `zotero_key` (8-char alphanumeric, e.g. `ABCD1234`) for cross-reference. DOI/ISBN stored in Zotero for external DB lookups when needed.

| Store | Purpose | Backend |
|-------|---------|---------|
| `top_of_mind` | Active projects, current context | Chroma (fast vector retrieval) |
| `coherence` | Identity, beliefs, preferences w/ confidence scores | Elasticsearch (structured + various-texts) |
| `who_i_was` | Edit history of above | Elasticsearch (temporal queries) |
| `store` | Everything relevant, 10:1 compressions | Elasticsearch (bulk storage) |
| `forgotten_store` | Archived with forgetting-reason | Elasticsearch (cold storage) |

Two Elasticsearch dockers - one for coherence/store, one for who_i_was/forgotten_store
Index-level isolation between the two stores on each docker.

## Services

Docker services in `services/`. Manage all with:

```bash
./services/services.sh up       # Start all
./services/services.sh down     # Stop all
./services/services.sh status   # Health & ports
./services/services.sh backup   # Timestamped backup
./services/services.sh reset    # Wipe & start fresh
```

| Service | Port | Purpose |
|---------|------|---------|
| Chroma | 8000 | Vector DB for top_of_mind |
| Elasticsearch (×2) | 9201, 9200 | Structured storage |
| Zotero | 3001, 23119 | Reference manager (headless + API) |

See `docs/stack.md` for versions.

## Setup

### 1. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Services

Start the Docker services:

```bash
# Elasticsearch
cd services/elasticsearch && docker compose up -d

# ChromaDB
cd services/chroma && docker compose up -d

# Zotero
cd services/zotero && docker compose up -d
```

### 3. Zotero Plugin Setup

The `zotero-local-crud` plugin provides HTTP CRUD access to the local Zotero library. It's automatically installed via init scripts.

**First-time setup:**

1. Start Zotero:
   ```bash
   cd services/zotero && docker compose up -d
   ```

2. Wait ~60 seconds for Zotero to fully start, then verify:
   ```bash
   curl http://localhost:23119/local-crud/ping
   ```

**Development (after editing plugin code):**

When editing plugin code, do a full restart with profile reset:
```bash
docker compose exec zotero rm -rf /config/.zotero
docker compose restart
# Wait 60 seconds for full startup
```

See `services/zotero/zotero-local-crud/notes.md` for detailed development notes.

## MCP Server

The `mcp_server/` package exposes all stores via MCP (Model Context Protocol) for LLM tool use.

```bash
python -m mcp_server.server
```

**Tools available:** 27 tools across stores (CRUD for top_of_mind, coherence, store, zotero; read-only for who_i_was, forgotten).

**Embedding config:** Set `THALA_EMBEDDING_PROVIDER` (openai/ollama) and `THALA_EMBEDDING_MODEL`.

## LangChain Tools

The `langchain_tools/` package provides tools for LangChain 1.x agents:

- **`search_memory`**: Cross-store semantic search (top_of_mind, coherence, store, who_i_was)
- **`expand_context`**: Deep retrieval by UUID, zotero_key, or content snippet
- **`process_document`**: Document extraction and summarization pipeline

```python
from langchain.agents import create_agent
from langchain_tools import search_memory, expand_context, process_document

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[search_memory, expand_context, process_document],
)
```

## Document Processing Workflow

The `workflows/document_processing/` package provides a LangGraph-based document processing pipeline using Anthropic Claude models:

- **Sonnet**: Standard summarization and metadata extraction
- **Opus + extended thinking**: Complex chapter analysis

```python
from workflows.document_processing import process_document, process_documents_with_batch_api

# Single document
result = await process_document("/path/to/doc.pdf")

# Multiple documents with 50% cost savings via Batch API
results = await process_documents_with_batch_api([
    {"id": "doc1", "content": "...", "title": "Doc 1"},
    {"id": "doc2", "content": "...", "title": "Doc 2"},
])
```