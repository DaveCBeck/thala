"""
Async Elasticsearch wrapper for structured storage.

Uses elasticsearch[async] 8.17.0 with native async support.
Routes to different ES instances based on index.

Index structure:
- ES Coherence (9201): store_l0, store_l1, store_l2, coherence
- ES Forgotten (9200): who_i_was, forgotten
"""

import logging
from typing import Any

from elasticsearch import AsyncElasticsearch

from .stores import CoherenceStore, ForgottenStore, MainStore, WhoIWasStore

logger = logging.getLogger(__name__)


class ElasticsearchStores:
    """
    Manages all Elasticsearch-backed stores.

    Routes operations to correct ES instance:
    - Port 9201: store_l0, store_l1, store_l2, coherence
    - Port 9200: who_i_was, forgotten
    """

    def __init__(
        self,
        coherence_host: str = "http://localhost:9201",
        forgotten_host: str = "http://localhost:9200",
        request_timeout: int = 30,
    ):
        # ES instance for coherence and store indices
        self._coherence_client = AsyncElasticsearch(
            hosts=[coherence_host],
            request_timeout=request_timeout,
            max_retries=3,
            retry_on_timeout=True,
            http_compress=True,
        )

        # ES instance for forgotten and who_i_was
        self._forgotten_client = AsyncElasticsearch(
            hosts=[forgotten_host],
            request_timeout=request_timeout,
            max_retries=3,
            retry_on_timeout=True,
            http_compress=True,
        )

        # Store instances
        self.coherence = CoherenceStore(self._coherence_client, self)
        self.store = MainStore(self._coherence_client, self)
        self.who_i_was = WhoIWasStore(self._forgotten_client)
        self.forgotten = ForgottenStore(self._forgotten_client)

    async def close(self):
        """Close all client connections."""
        await self._coherence_client.close()
        await self._forgotten_client.close()

    async def health_check(self) -> dict[str, Any]:
        """Check health of all ES instances."""
        try:
            coherence = await self._coherence_client.cluster.health()
            forgotten = await self._forgotten_client.cluster.health()
            return {
                "coherence_instance": coherence["status"],
                "forgotten_instance": forgotten["status"],
                "healthy": all(
                    s in ("green", "yellow")
                    for s in (coherence["status"], forgotten["status"])
                ),
            }
        except Exception as e:
            logger.error(f"ES health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
