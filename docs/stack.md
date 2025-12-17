# Project Stack

Technology stack and version constraints for research agents.

## Services (Docker)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Chroma | chromadb/chroma:1.0.0 | 8000 | Vector database |
| Elasticsearch (coherence) | elasticsearch:9.2.2 | 9201 | coherence, store indices |
| Elasticsearch (forgotten) | elasticsearch:9.2.2 | 9200 | who_i_was, forgotten_store indices |
| Qdrant | qdrant/qdrant:1.15.5 | 6333-6334 | Vector database |
| Zotero | linuxserver/zotero:latest | 3001, 23119 | Reference manager (headless) |

## Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | >=1.2.0 | LangChain agent framework |
| pydantic | >=2.0 | Data validation |
| chromadb | >=1.0.0 | ChromaDB Python client |
| elasticsearch[async] | >=8.17.0 | Elasticsearch async client |
| httpx | >=0.25.0 | Async HTTP client |
| mcp | >=1.0.0 | Model Context Protocol |

## Target Environment

- **Runtime**: Docker (WSL2)
- **OS**: Linux (Ubuntu/Debian on WSL2)
