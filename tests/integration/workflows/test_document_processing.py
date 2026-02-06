"""
Pytest tests for the document_processing workflow.

Tests:
1. Markdown file processing - uses smart_chunker path
2. PDF file processing - uses mocked Marker service

Uses test fixtures to isolate from local infrastructure:
- test_store_manager: StoreManager wired to testcontainers + mocks
- mock_marker: Mocked PDF processing (no real Marker service needed)
- mock_zotero: Mocked Zotero (no real Zotero service needed)
- configure_llm_broker_fast_mode: LLM broker in fast mode (direct calls, no batching)

Real LLM calls are made through the broker but executed directly (not batched).

Usage:
    pytest tests/integration/workflows/test_document_processing.py -m integration --quality quick
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from tests.factories import make_academic_paper_content

if TYPE_CHECKING:
    from langchain_tools.base import StoreManager

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def markdown_content() -> str:
    """Generate test markdown content for document processing."""
    return make_academic_paper_content(
        title="How to Do Great Work",
        sections=["Abstract", "Introduction", "Finding Your Work", "Working Hard", "Conclusion"],
    )


@pytest_asyncio.fixture
async def pdf_bytes() -> bytes:
    """Generate mock PDF bytes for testing."""
    # Simple mock PDF content - the mock_marker fixture will process this
    return b"%PDF-1.4\n%Mock PDF content for testing document processing workflow"


@pytest.fixture
def markdown_file(tmp_path: Path, markdown_content: str) -> Path:
    """Create a temporary markdown file for testing."""
    md_file = tmp_path / "test_essay.md"
    md_file.write_text(markdown_content)
    return md_file


@pytest.fixture
def pdf_file(tmp_path: Path, pdf_bytes: bytes) -> Path:
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path


def log_result_summary(result: dict, test_name: str) -> None:
    """Log a summary of the workflow result."""
    logger.info(f"=== TEST RESULT: {test_name} ===")

    status = result.get("current_status", "unknown")
    logger.info(f"Status: {status}")

    zotero_key = result.get("zotero_key")
    if zotero_key:
        logger.info(f"Zotero Key: {zotero_key}")

    proc_result = result.get("processing_result", {})
    if proc_result:
        logger.info(f"Word count: {proc_result.get('word_count', 'N/A')}")
        logger.info(f"Page count: {proc_result.get('page_count', 'N/A')}")
        logger.info(f"Chunks: {len(proc_result.get('chunks', []))}")

    short_summary = result.get("short_summary")
    if short_summary:
        logger.info(f"Short Summary ({len(short_summary)} chars): {short_summary[:200]}...")

    errors = result.get("errors", [])
    if errors:
        for error in errors:
            logger.error(f"Error: {error}")


@pytest.mark.integration
@pytest.mark.slow
class TestDocumentProcessing:
    """Integration tests for document processing workflow."""

    async def test_markdown_processing(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
        markdown_file: Path,
    ) -> None:
        """Test workflow with markdown file."""
        from workflows.document_processing import process_document

        logger.info(f"Testing markdown workflow with: {markdown_file}")

        # The test_store_manager fixture configures the global StoreManager singleton
        # with testcontainers and mocks, so workflows use it automatically
        result = await process_document(
            source=str(markdown_file),
            title="How to Do Great Work",
            item_type="blogPost",
            extra_metadata={
                "creators": [
                    {"creatorType": "author", "firstName": "Test", "lastName": "Author"}
                ],
                "date": "2024",
                "url": "https://example.com/test-essay",
            },
        )

        log_result_summary(result, "Markdown Processing")

        # Assertions
        assert result.get("current_status") == "completed", f"Expected completed, got {result.get('current_status')}"
        assert result.get("zotero_key") is not None, "Should have Zotero key"
        assert result.get("short_summary"), "Should have short summary"

        proc_result = result.get("processing_result", {})
        assert proc_result.get("word_count", 0) > 0, "Should have word count"
        assert len(proc_result.get("chunks", [])) > 0, "Should have chunks"

    async def test_pdf_processing(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
        pdf_file: Path,
        mock_marker: None,  # Ensures marker is mocked
    ) -> None:
        """Test workflow with PDF file (using mocked Marker)."""
        from workflows.document_processing import process_document

        logger.info(f"Testing PDF workflow with: {pdf_file}")
        # test_store_manager fixture configures the global StoreManager singleton

        result = await process_document(
            source=str(pdf_file),
            title="Test PDF Document",
            item_type="book",
            extra_metadata={
                "creators": [
                    {"creatorType": "author", "firstName": "Test", "lastName": "Writer"}
                ],
                "date": "2024",
                "publisher": "Test Publisher",
            },
        )

        log_result_summary(result, "PDF Processing")

        # Assertions
        assert result.get("current_status") == "completed", f"Expected completed, got {result.get('current_status')}"
        assert result.get("zotero_key") is not None, "Should have Zotero key"
        assert result.get("short_summary"), "Should have short summary"

        proc_result = result.get("processing_result", {})
        assert proc_result.get("word_count", 0) > 0, "Should have word count"


@pytest.mark.integration
@pytest.mark.slow
async def test_markdown_document_processing_standalone(
    test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
    markdown_file: Path,
) -> None:
    """Standalone test for markdown document processing.

    This test can be run individually without the class context.
    """
    from workflows.document_processing import process_document

    # test_store_manager fixture configures the global StoreManager singleton
    result = await process_document(
        source=str(markdown_file),
        title="Standalone Test Document",
        item_type="webpage",
    )

    assert result.get("current_status") == "completed"
    assert result.get("short_summary") is not None
