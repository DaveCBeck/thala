# Project Stack

Technology stack and version constraints for research agents.

## Services (Docker)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Chroma | chromadb/chroma:1.0.0 | 8000 | Vector database |
| Elasticsearch (coherence) | elasticsearch:9.2.2 | 9201 | coherence, store_l0, store_l1, store_l2 |
| Elasticsearch (forgotten) | elasticsearch:9.2.2 | 9200 | who_i_was, forgotten |
| Marker | marker-gpu | 8001 | GPU-accelerated PDF processing |
| Qdrant | qdrant/qdrant:1.15.5 | 6333-6334 | Vector database |
| Translation Server | zuphilip/translation-server | 1969 | Zotero bibliographic metadata extraction |
| Zotero | linuxserver/zotero:latest | 3001, 23119 | Reference manager (headless) |

## Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | >=1.2.6 | LangChain agent framework |
| langchain-anthropic | >=1.3.1 | Anthropic Claude integration |
| anthropic | >=0.40.0 | Anthropic API client |
| langgraph | >=1.0.6 | Graph-based workflows |
| pydantic | >=2.0 | Data validation |
| chromadb | >=1.0.0 | ChromaDB Python client |
| elasticsearch[async] | >=8.17.0 | Elasticsearch async client |
| httpx | >=0.25.0 | Async HTTP client |
| mcp | >=1.0.0 | Model Context Protocol |
| firecrawl-py | >=1.0.0 | Web scraping API |
| playwright | >=1.40.0 | Browser automation (scraper fallback) |
| html2text | >=2024.2.26 | HTML to markdown conversion |

## Target Environment

- **Runtime**: Docker (WSL2)
- **OS**: Linux (Ubuntu/Debian on WSL2)
