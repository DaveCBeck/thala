# Core Utilities

Async HTTP client infrastructure, caching decorators, and error handling utilities for service integrations.

## Usage

### Async HTTP Client

```python
from core.utils import BaseAsyncHttpClient, safe_http_request

class MyServiceClient(BaseAsyncHttpClient):
    def __init__(self):
        super().__init__(
            host_env_var="MY_SERVICE_HOST",
            port_env_var="MY_SERVICE_PORT",
            host_default="localhost",
            port_default=8080,
            timeout=30.0
        )

    async def fetch_data(self, resource_id: str) -> dict:
        client = await self._get_client()
        response = await safe_http_request(
            client,
            "GET",
            f"/api/resource/{resource_id}",
            error_class=RuntimeError
        )
        return response.json()

# Context manager usage
async with MyServiceClient() as service:
    data = await service.fetch_data("123")

# Manual cleanup
service = MyServiceClient()
try:
    data = await service.fetch_data("123")
finally:
    await service.close()
```

### Caching

```python
from core.utils import async_cached, generate_cache_key

# Simple caching with default key generation
@async_cached(namespace="my_service", ttl_days=7)
async def fetch_expensive_data(user_id: str) -> dict:
    # Expensive operation
    return {"user": user_id, "data": "..."}

# Custom cache key function
def user_cache_key(user_id: str, include_metadata: bool = False) -> str:
    return generate_cache_key("user", user_id, str(include_metadata))

@async_cached(namespace="users", ttl_days=1, key_fn=user_cache_key)
async def get_user_profile(user_id: str, include_metadata: bool = False) -> dict:
    return {"id": user_id, "metadata": include_metadata}
```

### Error Handling

```python
from core.utils import safe_http_request
import httpx

class MyServiceError(Exception):
    pass

async def call_api(client: httpx.AsyncClient):
    # Automatically handles ConnectError, TimeoutException, HTTPStatusError
    response = await safe_http_request(
        client,
        "POST",
        "/api/endpoint",
        error_class=MyServiceError,
        json={"key": "value"}
    )
    return response.json()
```

### Global Cleanup

```python
from core.utils import register_cleanup, cleanup_all_clients

# Register cleanup for application shutdown
client = MyServiceClient()
register_cleanup("my_service", client.close)

# On shutdown, close all registered clients
await cleanup_all_clients()
```

## Components

| Component | Purpose |
|-----------|---------|
| `BaseAsyncHttpClient` | Base class for async HTTP clients with lazy initialization and context manager support |
| `AsyncContextManager` | Base class for async context managers with `close()` method pattern |
| `async_cached` | Decorator for caching async function results with configurable TTL |
| `generate_cache_key` | SHA256-based cache key generation from string parts |
| `safe_http_request` | Consistent error handling wrapper for httpx requests |
| `register_cleanup` | Register cleanup functions for application shutdown |
| `cleanup_all_clients` | Execute all registered cleanup functions (idempotent) |

## Key Features

### Lazy Client Initialization
HTTP clients are initialized on first use, not at construction time. Supports automatic reconnection if client is closed.

### Environment-Based Configuration
Clients read host/port from environment variables with sensible defaults:
```python
BaseAsyncHttpClient(
    host_env_var="SERVICE_HOST",    # Falls back to host_default
    port_env_var="SERVICE_PORT",    # Falls back to port_default
    host_default="localhost",
    port_default=8000
)
```

### Persistent Caching
Cache decorator integrates with `workflows.shared.persistent_cache` for disk-based caching across runs. Disable caching via `CACHE_DISABLED` environment variable.

### Unified Error Handling
`safe_http_request` catches and converts httpx exceptions to custom error classes with structured logging:
- `ConnectError` → Connection failed
- `TimeoutException` → Request timeout
- `HTTPStatusError` → HTTP status code errors
- Generic exceptions → Unexpected errors

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CACHE_DISABLED` | Disable persistent caching | `false` |
| Custom `*_HOST` | Service host override | Defined per client |
| Custom `*_PORT` | Service port override | Defined per client |

## Related Modules

- `workflows.shared.persistent_cache` - Underlying cache storage implementation
- Service clients (e.g., `core/embedding/`, `core/ocr/`) - Consumers of `BaseAsyncHttpClient`
- Workflow coordinators - Use `cleanup_all_clients()` for graceful shutdown

## Architecture Notes

### Client Lifecycle
1. Instantiate client (no HTTP connection yet)
2. First method call triggers lazy initialization via `_get_client()`
3. Client persists across multiple calls
4. Explicit `close()` or context manager exit cleans up

### Cache Key Generation
Keys are SHA256 hashes to avoid filesystem path issues with complex arguments. Custom `key_fn` allows control over what constitutes cache identity.

### Cleanup Registry
Global registry pattern enables centralized cleanup without tight coupling. Services register their cleanup functions; coordinator calls `cleanup_all_clients()` once.
