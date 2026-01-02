"""Base async HTTP client with lazy initialization and context manager support."""

import os
from typing import Optional

import httpx

from .async_context import AsyncContextManager


class BaseAsyncHttpClient(AsyncContextManager):
    """
    Base async HTTP client with lazy initialization.

    Provides:
    - Lazy httpx.AsyncClient initialization
    - Environment variable configuration
    - Context manager support
    - Automatic cleanup
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
        base_url: Optional[str] = None,
        host_env_var: Optional[str] = None,
        port_env_var: Optional[str] = None,
        host_default: str = "localhost",
        port_default: int = 8000,
    ):
        if base_url:
            self.base_url = base_url
        else:
            self.host = host or os.environ.get(host_env_var or "", host_default)
            self.port = port or int(os.environ.get(port_env_var or "", str(port_default)))
            self.base_url = f"http://{self.host}:{self.port}"

        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
