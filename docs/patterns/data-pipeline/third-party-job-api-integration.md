---
name: third-party-job-api-integration
title: "Third-Party Service Integration with Job-Based API"
date: 2025-12-18
category: data-pipeline
applicability:
  - "When integrating services with long-running operations exceeding HTTP timeouts"
  - "When services require VPN or special network configuration"
  - "When variable completion times require status polling"
  - "When graceful degradation is needed when service unavailable"
components: [httpx, pydantic, asyncio, docker]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [async-client, job-polling, vpn-service, http-client, streaming-download, context-manager]
---

# Third-Party Service Integration with Job-Based API

## Intent

Provide a robust pattern for integrating external services that require long-running operations, network isolation (VPN), and job-based polling, with clean separation between service deployment and client implementation.

## Motivation

Some external service integrations cannot complete within typical HTTP request timeouts:

1. **Long-running operations**: Document retrieval, data processing, file conversions
2. **Network isolation**: Services requiring VPN connections to access institutional resources
3. **Variable completion times**: Operations that may take seconds or minutes
4. **Deployment complexity**: Private implementations that need to be isolated in submodules

This pattern addresses these challenges through:
- A three-layer architecture (service, API contract, client)
- Job submission and polling instead of synchronous requests
- Health checks that verify dependencies (VPN status)
- Streaming downloads for large results

## Applicability

Use this pattern when:
- Operations exceed typical HTTP timeouts (30s+)
- The service requires special network configuration (VPN, proxies)
- Completion times are unpredictable
- Results are files that need streaming download
- The implementation is private/sensitive (submodule isolation)

Do NOT use this pattern when:
- Operations complete quickly (<10s)
- Simple request/response is sufficient
- No special network requirements exist

## Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Three-Layer Architecture                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   Service Layer     │  Docker container with VPN networking
│   (submodule)       │  Private implementation
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   API Contract      │  RESTful endpoints (documented in README)
│   /health           │  - Health check with VPN status
│   /retrieve (POST)  │  - Submit job, returns job_id
│   /jobs/{id} (GET)  │  - Poll status
│   /jobs/{id}/file   │  - Download result
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Client Layer      │  Async Python client
│   (core/stores/)    │  - Context manager support
│                     │  - Polling helpers
│                     │  - Streaming downloads
└─────────────────────┘
```

## Implementation

### Step 1: Define Response Models

Use Pydantic models for type-safe API responses:

```python
from pydantic import BaseModel
from typing import Optional

class RetrieveJobResponse(BaseModel):
    """Response when a retrieval job is created."""
    job_id: str
    status: str


class RetrieveResult(BaseModel):
    """Result of a retrieval job."""
    job_id: str
    status: str  # pending, searching, downloading, completed, failed
    doi: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_format: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class HealthStatus(BaseModel):
    """Health check response with dependency status."""
    status: str
    vpn_connected: bool
    vpn_ip: Optional[str] = None
```

### Step 2: Create Base Async HTTP Client

A reusable base class with lazy initialization and cleanup:

```python
import httpx
import os
from typing import Optional

class BaseAsyncHttpClient:
    """Base async HTTP client with lazy initialization."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
        host_env_var: str = "SERVICE_HOST",
        port_env_var: str = "SERVICE_PORT",
        host_default: str = "localhost",
        port_default: int = 8000,
    ):
        self.host = host or os.environ.get(host_env_var, host_default)
        self.port = port or int(os.environ.get(port_env_var, str(port_default)))
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client (lazy initialization)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client (idempotent)."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
```

### Step 3: Implement Health Check with Dependency Status

Health checks should verify not just the service but its dependencies:

```python
async def health_check(self) -> bool:
    """Check if service is available and VPN is connected."""
    try:
        client = await self._get_client()
        response = await client.get("/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return data.get("vpn_connected", False)  # Check dependency
    except Exception as e:
        logger.debug(f"Health check failed: {e}")
    return False


async def get_health_status(self) -> Optional[HealthStatus]:
    """Get detailed health status for diagnostics."""
    try:
        client = await self._get_client()
        response = await client.get("/health", timeout=5.0)
        if response.status_code == 200:
            return HealthStatus.model_validate(response.json())
    except Exception as e:
        logger.debug(f"Health status check failed: {e}")
    return None
```

### Step 4: Implement Job Submission

Fire-and-forget job submission that returns immediately:

```python
async def retrieve(
    self,
    doi: str,
    title: Optional[str] = None,
    authors: Optional[list[str]] = None,
    timeout_seconds: int = 120,
) -> RetrieveJobResponse:
    """Submit a retrieval request."""
    client = await self._get_client()

    payload = {"doi": doi, "timeout_seconds": timeout_seconds}
    if title:
        payload["title"] = title
    if authors:
        payload["authors"] = authors

    response = await client.post("/retrieve", json=payload)
    response.raise_for_status()

    return RetrieveJobResponse.model_validate(response.json())
```

### Step 5: Implement Polling with Timeout

Wait for job completion with configurable timeout and poll interval:

```python
import asyncio

async def wait_for_completion(
    self,
    job_id: str,
    timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> RetrieveResult:
    """
    Wait for a job to complete.

    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    start_time = asyncio.get_event_loop().time()

    while True:
        result = await self.get_job_status(job_id)

        if result.status in ("completed", "failed"):
            return result

        # Check timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed >= timeout:
            raise asyncio.TimeoutError(
                f"Job {job_id} did not complete within {timeout}s"
            )

        await asyncio.sleep(poll_interval)


async def get_job_status(self, job_id: str) -> RetrieveResult:
    """Get the status of a retrieval job."""
    client = await self._get_client()
    response = await client.get(f"/jobs/{job_id}")
    response.raise_for_status()
    return RetrieveResult.model_validate(response.json())
```

### Step 6: Implement Streaming File Download

Stream large files to avoid memory issues:

```python
from pathlib import Path

async def download_file(self, job_id: str, local_path: str) -> str:
    """
    Download the retrieved file for a completed job.

    Uses streaming for potentially large files.
    """
    client = await self._get_client()

    async with client.stream("GET", f"/jobs/{job_id}/file") as response:
        response.raise_for_status()

        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)

        with open(local_file, "wb") as f:
            async for chunk in response.aiter_bytes():
                f.write(chunk)

    return str(local_file)
```

### Step 7: Provide Convenience Method

Combine submission, polling, and download into one call:

```python
async def retrieve_and_download(
    self,
    doi: str,
    local_path: str,
    title: Optional[str] = None,
    authors: Optional[list[str]] = None,
    timeout: float = 120.0,
) -> tuple[str, RetrieveResult]:
    """
    Convenience method: retrieve document and download to local path.

    Returns:
        Tuple of (local_path, result)

    Raises:
        asyncio.TimeoutError: If timeout exceeded
        httpx.HTTPStatusError: If retrieval fails
    """
    # Submit job
    job = await self.retrieve(doi=doi, title=title, authors=authors)

    # Poll for completion
    result = await self.wait_for_completion(job.job_id, timeout=timeout)

    if result.status == "failed":
        raise RuntimeError(f"Retrieval failed: {result.error_code}: {result.error}")

    # Download file
    await self.download_file(job.job_id, local_path)

    return local_path, result
```

### Step 8: Document API Contract

Create clear API contract documentation for alternative implementations:

```markdown
## API Contract

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check service health and VPN status |
| `/retrieve` | POST | Submit retrieval request (returns job_id) |
| `/jobs/{job_id}` | GET | Check job status and result |
| `/jobs/{job_id}/file` | GET | Download retrieved file |

### Job Status Values
- `pending` - Job queued
- `searching` - Searching for document
- `downloading` - Downloading document
- `completed` - File ready for download
- `failed` - Retrieval failed (check error_code)

### Error Codes
- `NOT_FOUND` - No matching document found
- `DOWNLOAD_FAILED` - Download failed after retries
- `VPN_ERROR` - VPN connection issue
- `TIMEOUT` - Operation timed out
```

## Usage Example

```python
from core.stores import RetrieveAcademicClient

async with RetrieveAcademicClient() as client:
    # Check if service is available
    if await client.health_check():
        # Retrieve and download in one call
        path, result = await client.retrieve_and_download(
            doi="10.1234/example",
            title="Paper Title",
            local_path="/tmp/paper.pdf",
        )
        print(f"Downloaded to {path}")
    else:
        print("Service unavailable or VPN disconnected")
```

## Consequences

### Benefits

- **Clean separation**: Service implementation isolated in submodule
- **Graceful degradation**: System works without VPN service if not configured
- **Type safety**: Pydantic models for all API responses
- **Streaming support**: Large file downloads use chunked streaming
- **Health checking**: VPN status validation before operations
- **Pluggable backends**: API contract allows alternative implementations

### Trade-offs

- **Polling overhead**: Multiple HTTP requests for long-running operations
- **Complexity**: Three-layer architecture is more complex than direct integration
- **Blocking file I/O**: `download_file` uses synchronous file writes (acceptable for moderate concurrency)
- **Timeout alignment**: Client and service timeouts need coordination

### Async Considerations

- Context manager ensures cleanup even on exceptions
- Polling uses `asyncio.sleep()` (non-blocking)
- Lazy client initialization avoids connection overhead
- Streaming download minimizes memory usage for large files

## Related Patterns

- [Parallel AI Search Integration](./parallel-ai-search-integration.md) - Uses similar async client patterns
- [Unified Scraping Service](./unified-scraping-service-fallback-chain.md) - Another external service integration
- [Citation Processing with Zotero](./citation-processing-zotero-integration.md) - Uses similar async client approach

## Known Uses in Thala

- `core/stores/retrieve_academic.py`: RetrieveAcademicClient implementation
- `core/utils/async_http_client.py`: BaseAsyncHttpClient base class
- `services/services.sh`: VPN_SERVICES array for conditional startup
- `services/README.md`: API contract documentation

## References

- [httpx Async Client](https://www.python-httpx.org/async/)
- [Pydantic v2 Models](https://docs.pydantic.dev/latest/)
- [asyncio Context Managers](https://docs.python.org/3/library/asyncio-task.html)
