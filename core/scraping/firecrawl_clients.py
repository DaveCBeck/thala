"""Firecrawl client management for local and cloud instances.

Provides lazy-initialized clients for both self-hosted and cloud Firecrawl,
with health checking for graceful degradation.
"""

import logging
from typing import TYPE_CHECKING, Optional

from .config import FirecrawlConfig, get_firecrawl_config

if TYPE_CHECKING:
    from firecrawl import AsyncFirecrawl

logger = logging.getLogger(__name__)


class FirecrawlClients:
    """Manager for local and cloud Firecrawl clients.

    Provides lazy initialization and maintains separate clients for:
    - Local (self-hosted): No API key needed
    - Cloud: Requires FIRECRAWL_API_KEY for stealth proxy access

    Usage:
        clients = get_firecrawl_clients()
        if clients.local:
            result = await clients.local.scrape(url)
    """

    def __init__(self, config: Optional[FirecrawlConfig] = None):
        self._config = config or get_firecrawl_config()
        self._local: "AsyncFirecrawl | None" = None
        self._cloud: "AsyncFirecrawl | None" = None

    def _get_local(self) -> "AsyncFirecrawl | None":
        """Get local Firecrawl client (lazy init)."""
        if not self._config.local_available:
            return None

        if self._local is None:
            from firecrawl import AsyncFirecrawl

            # Local instance - no API key validation by self-hosted server
            self._local = AsyncFirecrawl(
                api_key="local",  # Placeholder - not validated by self-hosted
                api_url=self._config.local_url,
            )
            logger.debug(f"Local Firecrawl client initialized: {self._config.local_url}")
        return self._local

    def _get_cloud(self) -> "AsyncFirecrawl | None":
        """Get cloud Firecrawl client (lazy init)."""
        if not self._config.cloud_available:
            return None

        if self._cloud is None:
            from firecrawl import AsyncFirecrawl

            self._cloud = AsyncFirecrawl(
                api_key=self._config.cloud_api_key,
                api_url=self._config.cloud_url,
            )
            logger.debug("Cloud Firecrawl client initialized")
        return self._cloud

    @property
    def local(self) -> "AsyncFirecrawl | None":
        """Get local client if available."""
        return self._get_local()

    @property
    def cloud(self) -> "AsyncFirecrawl | None":
        """Get cloud client if available."""
        return self._get_cloud()

    @property
    def config(self) -> FirecrawlConfig:
        """Get configuration."""
        return self._config

    async def close(self) -> None:
        """Close all client connections."""
        for client, name in [(self._local, "local"), (self._cloud, "cloud")]:
            if client is not None:
                try:
                    if hasattr(client, "close"):
                        await client.close()
                    elif hasattr(client, "_session") and client._session:
                        await client._session.close()
                except Exception as e:
                    logger.debug(f"Error closing {name} Firecrawl client: {e}")

        self._local = None
        self._cloud = None


# Module-level singleton
_clients: FirecrawlClients | None = None


async def close_firecrawl_clients() -> None:
    """Close the global FirecrawlClients instance."""
    global _clients
    if _clients is not None:
        await _clients.close()
        _clients = None


def get_firecrawl_clients() -> FirecrawlClients:
    """Get the global FirecrawlClients instance."""
    global _clients
    if _clients is None:
        from core.utils.async_http_client import register_cleanup

        _clients = FirecrawlClients()
        register_cleanup("FirecrawlClients", close_firecrawl_clients)
    return _clients
