"""
Pytest configuration for workflow tests.

This conftest provides fixtures and command-line options that work alongside
CLI-runnable tests. Tests can be run either via pytest or directly from command line.

Usage:
    # Via pytest with options
    pytest testing/ --quality quick --language en

    # Via pytest without options (uses defaults)
    pytest testing/test_cache.py

    # Direct CLI (unchanged behavior)
    python testing/test_research_workflow.py "topic" quick
"""

from collections.abc import Generator

import pytest

from core.logging import end_run, start_run


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


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy for async tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


# Mark slow tests for selective running
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (run with --runslow)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring external services",
    )
