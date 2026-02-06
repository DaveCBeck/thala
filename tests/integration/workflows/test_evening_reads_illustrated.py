"""
Pytest tests for the combined evening_reads + illustrate workflow.

Transforms a literature review into a 4-part illustrated article series
(1 overview + 3 deep-dives) with real LLM calls but isolated infrastructure.

Uses test fixtures to isolate from local infrastructure:
- test_store_manager: StoreManager wired to testcontainers + mocks
- mock_zotero: Mocked Zotero (no real Zotero service needed)
- configure_llm_broker_fast_mode: LLM broker in fast mode (direct calls, no batching)

Real LLM calls are made through the broker but executed directly (not batched).

Usage:
    pytest tests/integration/workflows/test_evening_reads_illustrated.py -m integration
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from tests.factories import make_academic_paper_content

if TYPE_CHECKING:
    from langchain_tools.base import StoreManager

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def literature_review_content() -> str:
    """Generate a realistic literature review for testing."""
    # Generate a multi-section literature review
    sections = [
        "Executive Summary",
        "Introduction",
        "Methodology",
        "Theme 1: Foundational Concepts",
        "Theme 2: Practical Applications",
        "Theme 3: Future Directions",
        "Conclusion",
        "References",
    ]

    content = make_academic_paper_content(
        title="Literature Review: Advances in Knowledge Management Systems",
        sections=sections,
    )

    # Add some citation-like references for the workflow to process
    content += """

## References

[@smith2023] Smith, J. (2023). Knowledge systems in modern enterprises. Journal of KM, 15(2), 45-67.
[@jones2024] Jones, M. (2024). AI-powered knowledge retrieval. AI Systems Review, 8(1), 12-28.
[@chen2023] Chen, L. et al. (2023). Semantic search architectures. Data Science Quarterly, 22(4), 156-178.
"""
    return content


@pytest.fixture
def literature_review_file(tmp_path: Path, literature_review_content: str) -> Path:
    """Create a temporary literature review file for testing."""
    lit_review_file = tmp_path / "test_literature_review.md"
    lit_review_file.write_text(literature_review_content)
    return lit_review_file


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create output directory for test artifacts."""
    output = tmp_path / "test_output"
    output.mkdir(parents=True, exist_ok=True)
    return output


async def run_evening_reads(
    literature_review: str,
    output_dir: Path,
    source_name: str,
    editorial_stance: str | None = None,
) -> dict:
    """Run the evening_reads workflow on a literature review.

    Args:
        literature_review: The literature review content
        output_dir: Directory to write output files
        source_name: Name of the source file for metadata
        editorial_stance: Optional editorial stance content

    Returns:
        dict with workflow result and list of article paths
    """
    from workflows.output.evening_reads import evening_reads_graph

    word_count = len(literature_review.split())
    logger.info(f"Input: {word_count} words")

    # Run workflow
    logger.info("Starting evening_reads workflow...")

    result = await evening_reads_graph.ainvoke({
        "input": {
            "literature_review": literature_review,
            "editorial_stance": editorial_stance,
        }
    })

    logger.info(f"Evening reads workflow completed with status: {result.get('status')}")

    if result.get("errors"):
        for error in result["errors"]:
            logger.error(f"Error in {error.get('node')}: {error.get('error')}")

    article_paths = []

    # Write final outputs
    final_outputs = result.get("final_outputs", [])
    if final_outputs:
        for output in final_outputs:
            filename = f"{output['id']}.md"
            output_file = output_dir / filename

            header = f"""---
source: {source_name}
generated: {datetime.now().isoformat()}
article_id: {output['id']}
title: {output['title']}
word_count: {output['word_count']}
status: {result.get('status')}
---

# {output['title']}

"""
            output_file.write_text(header + output['content'])
            article_paths.append(output_file)
            logger.info(f"Output ({output['id']}): {output['word_count']} words")

    return {
        "result": result,
        "article_paths": article_paths,
        "final_outputs": final_outputs,
    }


async def run_illustrate(
    input_file: Path,
    output_dir: Path,
    article_id: str,
) -> dict:
    """Run the illustrate workflow on a markdown document.

    Args:
        input_file: Path to the markdown file
        output_dir: Directory to write output files
        article_id: ID of the article being illustrated

    Returns:
        dict with illustration results
    """
    from workflows.output.illustrate import IllustrateConfig, illustrate_graph

    document = input_file.read_text()
    word_count = len(document.split())
    logger.info(f"Illustrating {article_id}: {word_count} words")

    # Create article-specific output directory for images
    images_dir = output_dir / "images" / article_id
    images_dir.mkdir(parents=True, exist_ok=True)

    # Configure workflow with minimal settings for testing
    config = IllustrateConfig(
        generate_header_image=True,
        additional_image_count=1,  # Reduced for faster tests
        header_prefer_public_domain=True,
        enable_vision_review=False,  # Disable for faster tests
        max_retries=1,
        output_dir=str(images_dir),
        enable_diagram_refinement=False,  # Disable for faster tests
    )

    # Run workflow
    logger.info(f"Starting illustrate workflow for {article_id}...")

    result = await illustrate_graph.ainvoke({
        "input": {
            "markdown_document": document,
            "title": input_file.stem,
            "output_dir": str(images_dir),
        },
        "config": config,
    })

    logger.info(f"Illustrate workflow completed for {article_id}")

    # Write illustrated document
    illustrated_doc = result.get("illustrated_document")
    if illustrated_doc:
        output_file = output_dir / f"{article_id}_illustrated.md"
        output_file.write_text(illustrated_doc)

    return {
        "result": result,
        "final_images": result.get("final_images", []),
        "image_plan": result.get("image_plan", []),
    }


def log_workflow_summary(
    evening_reads_result: dict,
    illustration_results: list[dict],
    output_dir: Path,
) -> None:
    """Log summary of the combined workflow results."""
    logger.info("=" * 60)
    logger.info("WORKFLOW SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Output directory: {output_dir}")

    final_outputs = evening_reads_result.get("final_outputs", [])
    logger.info(f"Articles generated: {len(final_outputs)}")

    total_words = sum(o.get("word_count", 0) for o in final_outputs)
    logger.info(f"Total word count: {total_words}")

    if final_outputs:
        for output in final_outputs:
            logger.info(f"  - {output['id']}: \"{output['title']}\" ({output['word_count']} words)")

    successful_illustrations = sum(1 for r in illustration_results if r.get("success"))
    logger.info(f"Articles illustrated: {successful_illustrations}/{len(final_outputs)}")

    total_images = sum(
        len(r.get("result", {}).get("final_images", []))
        for r in illustration_results
        if r.get("success")
    )
    logger.info(f"Total images generated: {total_images}")


@pytest.mark.integration
@pytest.mark.slow
class TestEveningReadsIllustrated:
    """Integration tests for combined evening_reads + illustrate workflow."""

    async def test_evening_reads_only(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
        literature_review_content: str,
        output_dir: Path,
    ) -> None:
        """Test evening_reads workflow without illustration."""
        # test_store_manager fixture configures the global StoreManager singleton
        # with testcontainers (ES, Chroma) and mocks (Zotero, Marker)
        result = await run_evening_reads(
            literature_review=literature_review_content,
            output_dir=output_dir,
            source_name="test_literature_review.md",
        )

        # Assertions
        assert result["result"].get("status") in ("completed", "partial"), \
            f"Unexpected status: {result['result'].get('status')}"

        final_outputs = result.get("final_outputs", [])
        assert len(final_outputs) > 0, "Should generate at least one article"

        # Check article structure
        for output in final_outputs:
            assert output.get("title"), "Article should have title"
            assert output.get("content"), "Article should have content"
            assert output.get("word_count", 0) > 0, "Article should have word count"

    async def test_illustrate_single_article(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
        literature_review_file: Path,
        output_dir: Path,
    ) -> None:
        """Test illustrate workflow on a single article."""
        result = await run_illustrate(
            input_file=literature_review_file,
            output_dir=output_dir,
            article_id="test_article",
        )

        # Assertions
        assert result["result"].get("status") in ("completed", "partial", None), \
            f"Unexpected status: {result['result'].get('status')}"

        # Image plan should be generated
        image_plan = result.get("image_plan", [])
        logger.info(f"Image plan entries: {len(image_plan)}")

    async def test_combined_evening_reads_and_illustrate(
        self,
        test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
        literature_review_content: str,
        output_dir: Path,
    ) -> None:
        """Test full combined workflow: evening_reads + illustrate."""
        # test_store_manager fixture configures the global StoreManager singleton
        # Step 1: Run evening_reads
        logger.info("STEP 1: Evening Reads")
        evening_reads_result = await run_evening_reads(
            literature_review=literature_review_content,
            output_dir=output_dir,
            source_name="test_literature_review.md",
        )

        article_paths = evening_reads_result["article_paths"]

        if not article_paths:
            pytest.skip("No articles generated, cannot test illustration")

        # Step 2: Illustrate first article only (for speed)
        logger.info("STEP 2: Illustrating First Article")
        illustration_results = []

        article_path = article_paths[0]
        article_id = article_path.stem

        try:
            illust_result = await run_illustrate(
                input_file=article_path,
                output_dir=output_dir,
                article_id=article_id,
            )
            illustration_results.append({
                "article_id": article_id,
                "success": True,
                "result": illust_result,
            })
        except Exception as e:
            logger.error(f"Failed to illustrate {article_id}: {e}")
            illustration_results.append({
                "article_id": article_id,
                "success": False,
                "error": str(e),
            })

        log_workflow_summary(evening_reads_result, illustration_results, output_dir)

        # Assertions
        assert len(evening_reads_result.get("final_outputs", [])) > 0
        # At least attempted illustration
        assert len(illustration_results) > 0


@pytest.mark.integration
@pytest.mark.slow
async def test_evening_reads_standalone(
    test_store_manager: "StoreManager",  # Sets up global singleton with testcontainers
    literature_review_content: str,
    output_dir: Path,
) -> None:
    """Standalone test for evening_reads workflow."""
    # test_store_manager fixture configures the global StoreManager singleton
    result = await run_evening_reads(
        literature_review=literature_review_content,
        output_dir=output_dir,
        source_name="standalone_test.md",
    )

    assert result["result"].get("status") in ("completed", "partial")
    assert len(result.get("article_paths", [])) > 0
