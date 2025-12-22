#!/usr/bin/env python3
"""
Test script for the document_processing workflow.

Tests:
1. Markdown file (Paul Graham essay) - uses smart_chunker path, no Marker needed
2. PDF file (War of the Worlds) - uses Marker path, requires Marker service running

Usage:
    python test_document_processing.py markdown   # Test markdown only
    python test_document_processing.py pdf        # Test PDF only (requires Marker)
    python test_document_processing.py all        # Test both
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
MARKDOWN_FILE = TEST_DATA_DIR / "paul_graham_great_work.md"
PDF_FILE = TEST_DATA_DIR / "war_of_the_worlds.pdf"


def print_result_summary(result: dict, test_name: str) -> None:
    """Print a summary of the workflow result."""
    print("\n" + "=" * 60)
    print(f"TEST RESULT: {test_name}")
    print("=" * 60)

    # Status
    status = result.get("current_status", "unknown")
    print(f"\nStatus: {status}")

    # Timing
    started = result.get("started_at")
    completed = result.get("completed_at")
    if started and completed:
        try:
            # Handle various datetime formats
            if isinstance(started, str):
                started = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if isinstance(completed, str):
                completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            # Make both naive for comparison
            if hasattr(started, 'replace') and started.tzinfo is not None:
                started = started.replace(tzinfo=None)
            if hasattr(completed, 'replace') and completed.tzinfo is not None:
                completed = completed.replace(tzinfo=None)
            duration = (completed - started).total_seconds()
            print(f"Duration: {duration:.1f}s")
        except Exception as e:
            print(f"Duration: (error calculating: {e})")

    # Zotero
    zotero_key = result.get("zotero_key")
    if zotero_key:
        print(f"Zotero Key: {zotero_key}")

    # Processing result
    proc_result = result.get("processing_result", {})
    if proc_result:
        print(f"\nProcessing Result:")
        print(f"  - Word count: {proc_result.get('word_count', 'N/A')}")
        print(f"  - Page count: {proc_result.get('page_count', 'N/A')}")
        print(f"  - Chunks: {len(proc_result.get('chunks', []))}")
        if proc_result.get('ocr_method'):
            print(f"  - OCR method: {proc_result.get('ocr_method')}")

    # Summary
    short_summary = result.get("short_summary")
    if short_summary:
        print(f"\nShort Summary ({len(short_summary)} chars):")
        # Show first 300 chars
        preview = short_summary[:300] + "..." if len(short_summary) > 300 else short_summary
        print(f"  {preview}")

    # Tenth summary (for large docs)
    tenth_summary = result.get("tenth_summary")
    if tenth_summary:
        print(f"\n10:1 Summary ({len(tenth_summary)} chars):")
        preview = tenth_summary[:300] + "..." if len(tenth_summary) > 300 else tenth_summary
        print(f"  {preview}")

    # Chapters
    chapters = result.get("chapters", [])
    if chapters:
        print(f"\nChapters detected: {len(chapters)}")
        for ch in chapters[:5]:  # Show first 5
            print(f"  - {ch.get('title', 'Untitled')}")
        if len(chapters) > 5:
            print(f"  ... and {len(chapters) - 5} more")

    # Chapter summaries
    chapter_summaries = result.get("chapter_summaries", [])
    if chapter_summaries:
        print(f"\nChapter summaries: {len(chapter_summaries)}")

    # Metadata
    metadata = result.get("metadata_updates", {})
    if metadata:
        print(f"\nExtracted Metadata:")
        for key, value in metadata.items():
            if value and key not in ("abstractNote",):  # Skip long fields
                print(f"  - {key}: {value}")

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - [{err.get('node', 'unknown')}]: {err.get('error', 'unknown')}")

    # Store records
    store_records = result.get("store_records", [])
    if store_records:
        print(f"\nStore records created: {len(store_records)}")

    print("\n" + "=" * 60)


async def test_markdown() -> dict:
    """Test workflow with markdown file (Paul Graham essay)."""
    from workflows.document_processing import process_document

    logger.info(f"Testing markdown workflow with: {MARKDOWN_FILE}")

    if not MARKDOWN_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {MARKDOWN_FILE}")

    result = await process_document(
        source=str(MARKDOWN_FILE),
        title="How to Do Great Work",
        item_type="blogPost",
        extra_metadata={
            "creators": [{"creatorType": "author", "firstName": "Paul", "lastName": "Graham"}],
            "date": "2023-07",
            "url": "https://paulgraham.com/greatwork.html",
        }
    )

    print_result_summary(result, "Markdown - Paul Graham Essay")
    return result


async def test_pdf() -> dict:
    """Test workflow with PDF file (War of the Worlds)."""
    from workflows.document_processing import process_document

    logger.info(f"Testing PDF workflow with: {PDF_FILE}")

    if not PDF_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {PDF_FILE}")

    result = await process_document(
        source=str(PDF_FILE),
        title="The War of the Worlds",
        item_type="book",
        quality="balanced",
        extra_metadata={
            "creators": [{"creatorType": "author", "firstName": "H.G.", "lastName": "Wells"}],
            "date": "1898",
            "publisher": "Project Gutenberg",
        }
    )

    print_result_summary(result, "PDF - War of the Worlds")
    return result


async def main():
    """Run document processing tests."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    test_type = sys.argv[1].lower()

    results = {}

    if test_type in ("markdown", "md", "all"):
        try:
            results["markdown"] = await test_markdown()
            logger.info("Markdown test completed successfully!")
        except Exception as e:
            logger.error(f"Markdown test failed: {e}", exc_info=True)
            results["markdown"] = {"error": str(e)}

    if test_type in ("pdf", "all"):
        try:
            results["pdf"] = await test_pdf()
            logger.info("PDF test completed successfully!")
        except Exception as e:
            logger.error(f"PDF test failed: {e}", exc_info=True)
            results["pdf"] = {"error": str(e)}

    if test_type not in ("markdown", "md", "pdf", "all"):
        print(f"Unknown test type: {test_type}")
        print("Use: markdown, pdf, or all")
        sys.exit(1)

    # Save results to file
    output_file = TEST_DATA_DIR / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
