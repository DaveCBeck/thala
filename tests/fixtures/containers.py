"""Testcontainer fixtures for isolated test infrastructure.

Provides session-scoped async fixtures for Elasticsearch and ChromaDB
that start containers with dynamic ports, avoiding conflicts with
production services.

Usage:
    @pytest.mark.integration
    async def test_es_operations(es_container):
        # es_container is the URL string, e.g., "http://localhost:49152"
        ...
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import TypedDict

import httpx
import pytest_asyncio
from testcontainers.elasticsearch import ElasticSearchContainer
from testcontainers.core.container import DockerContainer

logger = logging.getLogger(__name__)

# Match production versions
ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:9.0.0"
CHROMA_IMAGE = "chromadb/chroma:latest"


class ContainerConfig(TypedDict):
    """Type-safe container configuration."""

    es_url: str
    chroma_host: str
    chroma_port: int


@pytest_asyncio.fixture(scope="session")
async def es_container() -> AsyncGenerator[str, None]:
    """Session-scoped Elasticsearch testcontainer.

    Uses asyncio.to_thread() to avoid blocking the event loop during
    container startup.

    Yields:
        The Elasticsearch URL (e.g., "http://localhost:49152")

    Raises:
        TimeoutError: If container fails to become healthy within 120s
    """
    container = (
        ElasticSearchContainer(image=ES_IMAGE)
        .with_env("discovery.type", "single-node")
        .with_env("xpack.security.enabled", "false")
        .with_env("ES_JAVA_OPTS", "-Xms512m -Xmx512m")
    )

    logger.info("Starting Elasticsearch testcontainer...")
    await asyncio.to_thread(container.start)

    try:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(9200))
        url = f"http://{host}:{port}"

        # Health check using HTTP endpoint
        await _wait_for_http_ready(f"{url}/_cluster/health", timeout=120.0)
        logger.info(f"Elasticsearch testcontainer ready at {url}")
        yield url
    except Exception as e:
        logger.error(f"ES container health check failed: {e}")
        raise
    finally:
        logger.info("Stopping Elasticsearch testcontainer...")
        await asyncio.to_thread(container.stop)


@pytest_asyncio.fixture(scope="session")
async def chroma_container() -> AsyncGenerator[tuple[str, int], None]:
    """Session-scoped ChromaDB testcontainer.

    Yields:
        Tuple of (host, port) for ChromaDB HTTP client
    """
    container = (
        DockerContainer(CHROMA_IMAGE)
        .with_exposed_ports(8000)
        .with_env("ANONYMIZED_TELEMETRY", "false")
        .with_env("ALLOW_RESET", "true")
    )

    logger.info("Starting ChromaDB testcontainer...")
    await asyncio.to_thread(container.start)

    try:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(8000))

        # Async health check with httpx (ChromaDB 1.0+ uses v2 API)
        await _wait_for_http_ready(f"http://{host}:{port}/api/v2/heartbeat")
        logger.info(f"ChromaDB testcontainer ready at {host}:{port}")
        yield host, port
    except Exception as e:
        logger.error(f"ChromaDB container health check failed: {e}")
        raise
    finally:
        logger.info("Stopping ChromaDB testcontainer...")
        await asyncio.to_thread(container.stop)


async def _wait_for_http_ready(
    url: str,
    timeout: float = 60.0,
    interval: float = 0.5,
) -> None:
    """Wait for HTTP endpoint to become ready (async, non-blocking).

    Args:
        url: The URL to poll
        timeout: Maximum seconds to wait
        interval: Seconds between poll attempts

    Raises:
        TimeoutError: If endpoint not ready within timeout
    """
    deadline = time.monotonic() + timeout

    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    return
            except (
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ReadError,
                httpx.HTTPStatusError,
            ):
                # Container still starting up
                pass
            await asyncio.sleep(interval)

    raise TimeoutError(f"HTTP endpoint {url} not ready after {timeout}s")


@pytest_asyncio.fixture(scope="session")
async def containers(
    es_container: str,
    chroma_container: tuple[str, int],
) -> AsyncGenerator[ContainerConfig, None]:
    """Convenience fixture that ensures both containers are ready.

    pytest-asyncio will start es_container and chroma_container in parallel
    when they have no dependencies on each other.

    Yields:
        ContainerConfig with URLs/ports for all containers
    """
    yield ContainerConfig(
        es_url=es_container,
        chroma_host=chroma_container[0],
        chroma_port=chroma_container[1],
    )


@pytest_asyncio.fixture(scope="session")
async def es_with_indices(es_container: str) -> AsyncGenerator[str, None]:
    """ES container with thala indices created.

    Sets up all index templates and creates indices using the same
    setup_indices() function used in production.

    Yields:
        The Elasticsearch URL with indices ready
    """
    from core.stores.setup_indices import setup_indices

    logger.info("Setting up ES indices in testcontainer...")
    await setup_indices(
        reset=False,
        coherence_host=es_container,
        forgotten_host=es_container,  # Use same instance for tests
    )
    logger.info("ES indices ready")
    yield es_container
