# Thala Services

This directory contains Docker-based services that support the thala workflow system.

## Services Overview

| Service | Port | Description |
|---------|------|-------------|
| elasticsearch-coherence | 9201 | Primary ES instance for coherence and store indices |
| elasticsearch-forgotten | 9200 | ES instance for historical/archived content |
| chroma | 8000 | ChromaDB vector database for embeddings |
| zotero | 3001 (GUI), 23119 (API) | Zotero reference manager with local-crud API |
| marker | 8001 | GPU-accelerated PDF to Markdown processor |
| translation-server | 1969 | Bibliographic metadata extraction from URLs |
| retrieve-academic | 8002 | Academic document retrieval (VPN, submodule) |

## Quick Start

```bash
# Start all services
./services.sh up

# Check status
./services.sh status

# Stop all services
./services.sh down
```

## Service Categories

### Regular Services
- **elasticsearch-coherence**: Identity, beliefs, and main content storage
- **elasticsearch-forgotten**: Historical snapshots and archived content
- **chroma**: Vector embeddings for semantic search
- **zotero**: Citation and reference management

### GPU Services (require nvidia-container-toolkit)
- **marker**: Document processing with OCR

### VPN Services (require credentials and submodule init)
- **retrieve-academic**: Full-text academic document retrieval

## retrieve-academic Contract

The `retrieve-academic` service provides a pluggable interface for retrieving full-text academic documents when open access URLs are not available.

### Setup (Submodule)

This service is a **git submodule** pointing to a separate private repository:

```bash
# Initialize the submodule (after cloning thala)
git submodule update --init services/retrieve-academic

# Configure VPN credentials
cp services/retrieve-academic/.env.example services/retrieve-academic/.env
# Edit .env with your credentials
```

### API Contract

**Endpoint**: `POST /retrieve`

**Request**:
```json
{
  "doi": "10.1234/example",
  "title": "Optional Paper Title",
  "authors": ["Optional", "Author Names"],
  "preferred_formats": ["pdf", "epub"],
  "timeout_seconds": 120
}
```

**Response** (Job created):
```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

**Endpoint**: `GET /jobs/{job_id}`

**Response** (Completed):
```json
{
  "job_id": "uuid",
  "status": "completed",
  "doi": "10.1234/example",
  "file_path": "10.1234_example/abc123.pdf",
  "file_format": "pdf",
  "file_size": 1234567,
  "source_id": "abc123"
}
```

**Response** (Failed):
```json
{
  "job_id": "uuid",
  "status": "failed",
  "doi": "10.1234/example",
  "error": "Document not found",
  "error_code": "NOT_FOUND"
}
```

**Error Codes**:
- `NOT_FOUND`: No matching document found
- `DOWNLOAD_FAILED`: Download failed after retries
- `VPN_ERROR`: VPN connection issue
- `TIMEOUT`: Operation timed out

**Endpoint**: `GET /jobs/{job_id}/file`

Returns the downloaded file as binary stream.

### Implementing Alternative Backends

The API contract is designed to be implementation-agnostic. You can implement alternatives using:

1. **Institutional APIs**: If your institution provides API access to journals
2. **Browser Automation**: Use Playwright/Selenium with institutional credentials
3. **Interlibrary Loan**: Queue requests to your ILL system

Requirements for alternative implementations:
- Expose the same HTTP endpoints
- Use the same request/response schemas
- Return the same error codes
- Store files in the `downloads/` volume

### Python Client

Use the async client from `core/stores/retrieve_academic.py`:

```python
from core.stores import RetrieveAcademicClient

async with RetrieveAcademicClient() as client:
    # Check if service is available
    if await client.health_check():
        # Retrieve and download
        path, result = await client.retrieve_and_download(
            doi="10.1234/example",
            title="Paper Title",
            local_path="/tmp/paper.pdf",
        )
```

## Backup and Restore

```bash
# Create backup of all service data
./services.sh backup

# Restore from backup
./services.sh restore backups/20251216-120000

# Reset all data (destructive!)
./services.sh reset
```

## Monitoring

When `THALA_MODE=dev` in `.env`, monitoring starts automatically with `services.sh up`:
- **Metrics**: Saved to `services/metrics/YYYY-MM-DD.jsonl`
- **Container logs**: Streamed to `services/logs/<container>-<timestamp>.log`

```bash
# Manual foreground monitoring (table output)
./services.sh monitor

# One-shot status check
./services.sh monitor --once

# Custom interval (60s instead of default 30s)
./services.sh monitor --interval 60
```

Metrics collected: CPU%, memory usage, health endpoint latency (avg, p90).

## Logs

```bash
# Follow logs for a specific service
./services.sh logs zotero
./services.sh logs marker
```

In dev mode, container logs are also captured to `services/logs/` on startup.
