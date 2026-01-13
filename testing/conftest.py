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

import pytest


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
