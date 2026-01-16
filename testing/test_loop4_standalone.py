#!/usr/bin/env python3
"""
Standalone test for Loop 4 (Section-Level Deep Editing).

This script runs Loop 4 in isolation using simulated Loop 3 output,
with enhanced logging to capture:
- Tool usage (search queries, results, paper content fetched)
- Section editing details
- Holistic review decisions

Usage:
    python testing/test_loop4_standalone.py [--input PATH] [--quality LEVEL]

Example:
    python testing/test_loop4_standalone.py --input testing/test_data/supervised_lit_review_loop3_20260114_090805.md
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Enable dev mode for LangSmith tracing
os.environ["THALA_MODE"] = "dev"

from testing.utils import (
    configure_logging,
    get_output_dir,
    save_json_result,
    save_markdown_report,
    print_section_header,
)

# Configure logging with DEBUG level for detailed output
configure_logging("test_loop4_standalone")

# Create a separate logger for tool calls
tool_logger = logging.getLogger("tool_calls")
tool_handler = logging.FileHandler(
    f"logs/loop4_tool_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
tool_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
tool_logger.addHandler(tool_handler)
tool_logger.setLevel(logging.DEBUG)

# Set all relevant loggers to DEBUG
for logger_name in [
    "workflows.wrappers.supervised_lit_review",
    "workflows.wrappers.supervised_lit_review.supervision",
    "workflows.wrappers.supervised_lit_review.supervision.loops",
    "workflows.wrappers.supervised_lit_review.supervision.loops.loop4_editing",
    "workflows.wrappers.supervised_lit_review.supervision.tools",
    "workflows.shared.llm_utils",
]:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

OUTPUT_DIR = get_output_dir()

# Quality presets
QUALITY_PRESETS = {
    "test": {"max_stages": 1},
    "quick": {"max_stages": 2},
    "standard": {"max_stages": 3},
    "comprehensive": {"max_stages": 4},
}


def monkey_patch_tools_for_logging():
    """Patch paper search tools to log all calls and responses."""
    from workflows.wrappers.supervised_lit_review.supervision.tools.paper_search import searcher

    original_create_paper_tools = searcher.create_paper_tools

    def logged_create_paper_tools(store_query):
        """Create paper tools with logging wrappers."""
        tools = original_create_paper_tools(store_query)

        # Find the search_papers and get_paper_content tools
        search_tool = None
        content_tool = None
        for tool in tools:
            if tool.name == "search_papers":
                search_tool = tool
            elif tool.name == "get_paper_content":
                content_tool = tool

        if search_tool:
            original_search = search_tool.coroutine
            async def logged_search(query: str, limit: int = 10):
                tool_logger.info(f"SEARCH PAPERS: query='{query}', limit={limit}")
                result = await original_search(query, limit)
                tool_logger.info(f"SEARCH RESULTS: {result.get('total_found', 0)} papers found")
                for paper in result.get("papers", [])[:5]:
                    tool_logger.debug(
                        f"  - [{paper.get('zotero_key')}] {paper.get('title', '')[:60]}... "
                        f"(relevance: {paper.get('relevance', 0):.3f})"
                    )
                return result
            search_tool.coroutine = logged_search

        if content_tool:
            original_content = content_tool.coroutine
            async def logged_content(zotero_key: str, max_chars: int = 10000):
                tool_logger.info(f"GET PAPER CONTENT: key={zotero_key}, max_chars={max_chars}")
                result = await original_content(zotero_key, max_chars)
                content_len = len(result.get("content", ""))
                tool_logger.info(
                    f"CONTENT RETURNED: {content_len} chars for '{result.get('title', 'Unknown')[:50]}'"
                )
                tool_logger.debug(f"  First 500 chars: {result.get('content', '')[:500]}...")
                return result
            content_tool.coroutine = logged_content

        return tools

    searcher.create_paper_tools = logged_create_paper_tools
    logger.info("Monkey-patched paper search tools for detailed logging")


async def run_loop4_test(
    input_file: str,
    quality: str = "test",
) -> dict:
    """Run Loop 4 standalone test.

    Args:
        input_file: Path to Loop 3 output markdown file
        quality: Quality preset (test, quick, standard, comprehensive)

    Returns:
        Dict with test results and analysis
    """
    from workflows.wrappers.supervised_lit_review.supervision.loops.loop4_editing import (
        run_loop4_standalone,
    )

    # Read input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    review_text = input_path.read_text()
    logger.info(f"Loaded input file: {input_file} ({len(review_text)} chars)")

    # Extract topic from the file (assuming standard format)
    topic = "Attention mechanisms in transformer architectures"  # Default
    for line in review_text.split("\n")[:5]:
        if line.startswith("# Literature Review"):
            topic = line.replace("# Literature Review:", "").replace("# Literature Review", "").strip()
            break

    logger.info(f"Topic: {topic}")

    # Quality settings
    quality_settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["test"])
    logger.info(f"Quality settings: {quality_settings}")

    # Patch tools for logging
    monkey_patch_tools_for_logging()

    # Run Loop 4
    print_section_header("RUNNING LOOP 4")
    start_time = datetime.now()

    result = await run_loop4_standalone(
        review=review_text,
        topic=topic,
        quality_settings=quality_settings,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"Loop 4 completed in {duration:.1f}s")
    logger.info(f"Iterations used: {result.iterations_used}")
    logger.info(f"Changes summary: {result.changes_summary}")

    # Prepare output
    test_result = {
        "input_file": str(input_file),
        "topic": topic,
        "quality": quality,
        "quality_settings": quality_settings,
        "duration_seconds": duration,
        "iterations_used": result.iterations_used,
        "changes_summary": result.changes_summary,
        "input_length_chars": len(review_text),
        "output_length_chars": len(result.current_review),
        "input_word_count": len(review_text.split()),
        "output_word_count": len(result.current_review.split()),
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
    }

    return {
        "result": test_result,
        "output_review": result.current_review,
    }


def print_test_summary(test_result: dict):
    """Print test result summary."""
    print_section_header("TEST RESULTS")
    r = test_result["result"]

    print(f"\nInput: {r['input_file']}")
    print(f"Topic: {r['topic']}")
    print(f"Quality: {r['quality']}")
    print(f"\nDuration: {r['duration_seconds']:.1f}s")
    print(f"Iterations used: {r['iterations_used']}")
    print(f"\nInput: {r['input_word_count']} words ({r['input_length_chars']} chars)")
    print(f"Output: {r['output_word_count']} words ({r['output_length_chars']} chars)")

    word_change = r['output_word_count'] - r['input_word_count']
    pct_change = (word_change / r['input_word_count']) * 100 if r['input_word_count'] > 0 else 0
    print(f"Change: {word_change:+d} words ({pct_change:+.1f}%)")

    print(f"\nChanges summary: {r['changes_summary']}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Loop 4 standalone test with detailed logging"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="testing/test_data/supervised_lit_review_loop3_20260114_090805.md",
        help="Path to Loop 3 output file",
    )

    parser.add_argument(
        "--quality",
        type=str,
        choices=["test", "quick", "standard", "comprehensive"],
        default="test",
        help="Quality preset (default: test)",
    )

    return parser.parse_args()


async def main():
    """Run the Loop 4 standalone test."""
    args = parse_args()

    print_section_header("LOOP 4 STANDALONE TEST")
    print(f"\nInput file: {args.input}")
    print(f"Quality: {args.quality}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        test_result = await run_loop4_test(
            input_file=args.input,
            quality=args.quality,
        )

        # Print summary
        print_test_summary(test_result)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save result JSON
        result_file = save_json_result(
            test_result["result"],
            f"loop4_test_result_{timestamp}",
        )
        logger.info(f"Result saved to: {result_file}")

        # Save output review
        output_file = save_markdown_report(
            test_result["output_review"],
            f"loop4_test_output_{timestamp}",
            title=f"Loop 4 Output: {test_result['result']['topic']}",
            metadata={
                "quality": args.quality,
                "iterations": test_result["result"]["iterations_used"],
                "duration_seconds": test_result["result"]["duration_seconds"],
            },
        )
        logger.info(f"Output review saved to: {output_file}")

        print(f"\nOutput files:")
        print(f"  Result: {result_file}")
        print(f"  Review: {output_file}")
        print(f"  Tool log: logs/loop4_tool_calls_*.log")

        return test_result

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

    finally:
        # Clean up HTTP clients
        from core.utils.async_http_client import cleanup_all_clients
        await cleanup_all_clients()


if __name__ == "__main__":
    asyncio.run(main())
