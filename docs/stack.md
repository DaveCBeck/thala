# Project Stack

Technology stack and version constraints for research agents.

## Services (Docker)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Chroma | chromadb/chroma:1.0.0 | 8000 | Vector database |
| Elasticsearch (coherence) | elasticsearch:8.17.0 | 9201 | coherence, store indices |
| Elasticsearch (forgotten) | elasticsearch:8.17.0 | 9202 | who_i_was, forgotten_store indices |
| Qdrant | qdrant/qdrant:1.15.5 | 6333-6334 | Vector database |
| Zotero | linuxserver/zotero:latest | 3001, 23119 | Reference manager (headless) |

## Target Environment

- **Runtime**: Docker (WSL2)
- **OS**: Linux (Ubuntu/Debian on WSL2)
