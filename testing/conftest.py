"""
Pytest configuration for thala tests.

This conftest provides:
- Test infrastructure fixtures (testcontainers, mocks)
- Command-line options for workflow tests
- Logging isolation per test module

Usage:
    # Run unit tests only
    pytest testing/ -m unit

    # Run integration tests (uses testcontainers)
    pytest testing/ -m integration

    # Run with quality level for workflow tests
    pytest testing/ --quality quick --language en

    # Run with parallel workers (uses --dist loadscope for container efficiency)
    pytest testing/ -n auto --dist loadscope
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from core.logging import end_run, start_run

# Re-export fixtures from testing/fixtures/ for pytest discovery
from testing.fixtures import (
    ContainerConfig,
    chroma_container,
    containers,
    es_container,
    es_with_indices,
    mock_marker,
    mock_zotero,
)

if TYPE_CHECKING:
    from langchain_tools.base import StoreManager

# Make fixtures available to pytest
__all__ = [
    "es_container",
    "chroma_container",
    "containers",
    "es_with_indices",
    "mock_zotero",
    "mock_marker",
    "ContainerConfig",
    "test_store_manager",
]


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async fixtures.

    This is required for session-scoped async fixtures to work properly
    with pytest-asyncio.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def logging_run(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Rotate logs at test module boundaries.

    Each test module gets its own logging run, which triggers log rotation
    on first write to each module's log file.

    When running with pytest-xdist, each worker uses a separate log directory
    to prevent file corruption from concurrent writes.
    """
    import os

    # Handle pytest-xdist worker isolation to prevent log file corruption
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        os.environ["THALA_LOG_DIR"] = f"logs/test-{worker_id}"

    # Use test module path as run identifier (e.g., "test-testing-test_cache")
    test_path = request.node.nodeid.split("::")[0]  # e.g., "testing/test_cache.py"
    test_name = test_path.replace("/", "-").replace(".py", "")
    start_run(f"test-{test_name}")
    yield
    end_run()


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--quality",
        action="store",
        default="quick",
        choices=["test", "quick", "standard", "comprehensive", "high_quality"],
        help="Quality level for workflow tests (default: quick)",
    )
    parser.addoption(
        "--language",
        action="store",
        default="en",
        help="Language code for workflow tests (default: en)",
    )


@pytest.fixture
def quality_level(request):
    """Get the quality level from pytest options."""
    return request.config.getoption("--quality")


@pytest.fixture
def language(request):
    """Get the language code from pytest options."""
    return request.config.getoption("--language")


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory for test results."""
    output = tmp_path / "test_output"
    output.mkdir()
    return output


def pytest_configure(config):
    """Configure custom markers.

    Note: These are also defined in pyproject.toml, but we keep them here
    for backwards compatibility and explicit documentation.
    """
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (no external services)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (>30s)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (uses testcontainers)",
    )


@pytest_asyncio.fixture
async def test_store_manager(
    es_with_indices: str,
    chroma_container: tuple[str, int],
    mock_zotero: AsyncMock,
    mock_marker: None,
) -> AsyncGenerator["StoreManager", None]:
    """StoreManager wired to testcontainers and mocks.

    This fixture provides a fully configured StoreManager that uses:
    - Elasticsearch testcontainer (with indices)
    - ChromaDB testcontainer
    - Mocked Zotero (no real Zotero service needed)
    - Mocked Marker (no real Marker service needed)

    Yields:
        Configured StoreManager with test infrastructure

    Note:
        Directly sets manager._zotero for test injection. This is intentional
        test-only access to inject the mock without modifying production code.
    """
    from langchain_tools.base import StoreManager, _reset_default_manager

    # Use the test helper for clean singleton reset
    _reset_default_manager()

    chroma_host, chroma_port = chroma_container

    manager = StoreManager(
        es_coherence_host=es_with_indices,
        es_forgotten_host=es_with_indices,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
    )

    # Inject mock zotero (intentional test-only private attribute access)
    manager._zotero = mock_zotero

    yield manager

    await manager.close()
    _reset_default_manager()
