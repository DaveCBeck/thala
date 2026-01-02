"""Per-cluster deep analysis implementation.

Uses Anthropic Batch API for 50% cost reduction when analyzing 5+ clusters.
"""

import asyncio
import json
import logging
from typing import Any
from typing_extensions import TypedDict

from workflows.research.subgraphs.academic_lit_review.state import ThematicCluster
from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.shared.batch_processor import BatchProcessor

from .prompts import CLUSTER_ANALYSIS_SYSTEM_PROMPT, CLUSTER_ANALYSIS_USER_TEMPLATE
from .schemas import ClusterAnalysisOutput

logger = logging.getLogger(__name__)


class ClusterAnalysis(TypedDict):
    """Deep analysis of a single thematic cluster."""

    cluster_id: int
    narrative_summary: str  # 2-3 paragraph summary of the theme
    timeline: list[str]  # Key developments chronologically
    key_debates: list[str]  # Main debates and positions
    methodologies: list[str]  # Common methodological approaches
    outstanding_questions: list[str]  # Open research questions


async def per_cluster_analysis_node(state: dict) -> dict[str, Any]:
    """Generate deep analysis for each thematic cluster.

    Uses Anthropic Batch API for 50% cost reduction when analyzing 5+ clusters.
    """
    final_clusters = state.get("final_clusters", [])
    paper_summaries = state.get("paper_summaries", {})

    if not final_clusters:
        logger.warning("No clusters to analyze")
        return {"cluster_analyses": []}

    # Use batch API for 5+ clusters (50% cost reduction)
    if len(final_clusters) >= 5:
        return await _per_cluster_analysis_batched(final_clusters, paper_summaries)

    # Fall back to concurrent calls for small batches
    async def analyze_single_cluster(cluster: ThematicCluster) -> ClusterAnalysis:
        """Analyze a single cluster."""
        cluster_dois = cluster["paper_dois"]

        # Format papers for analysis
        papers_detail = []
        for doi in cluster_dois:
            summary = paper_summaries.get(doi)
            if not summary:
                continue

            paper_text = f"""
Title: {summary.get('title', 'Unknown')}
Year: {summary.get('year', 'Unknown')}
Summary: {summary.get('short_summary', 'N/A')}
Key Findings: {'; '.join(summary.get('key_findings', [])[:3])}
Methodology: {summary.get('methodology', 'N/A')[:150]}
Limitations: {'; '.join(summary.get('limitations', [])[:2])}"""
            papers_detail.append(paper_text)

        papers_text = "\n---\n".join(papers_detail)

        user_prompt = CLUSTER_ANALYSIS_USER_TEMPLATE.format(
            theme_name=cluster["label"],
            theme_description=cluster["description"],
            papers_detail=papers_text,
        )

        try:
            llm = get_llm(tier=ModelTier.SONNET, max_tokens=4096)

            # Use structured output to avoid JSON parsing issues
            structured_llm = llm.with_structured_output(ClusterAnalysisOutput)
            messages = [
                {"role": "system", "content": CLUSTER_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            result: ClusterAnalysisOutput = await structured_llm.ainvoke(messages)

            return ClusterAnalysis(
                cluster_id=cluster["cluster_id"],
                narrative_summary=result.narrative_summary,
                timeline=result.timeline,
                key_debates=result.key_debates,
                methodologies=result.methodologies,
                outstanding_questions=result.outstanding_questions,
            )

        except Exception as e:
            logger.warning(f"Failed to analyze cluster {cluster['label']}: {e}")
            return ClusterAnalysis(
                cluster_id=cluster["cluster_id"],
                narrative_summary=f"Analysis failed: {e}",
                timeline=[],
                key_debates=[],
                methodologies=[],
                outstanding_questions=[],
            )

    semaphore = asyncio.Semaphore(5)

    async def analyze_with_limit(cluster: ThematicCluster) -> ClusterAnalysis:
        async with semaphore:
            return await analyze_single_cluster(cluster)

    tasks = [analyze_with_limit(c) for c in final_clusters]
    analyses = await asyncio.gather(*tasks, return_exceptions=True)

    cluster_analyses: list[ClusterAnalysis] = []
    for result in analyses:
        if isinstance(result, Exception):
            logger.error(f"Cluster analysis task failed: {result}")
            continue
        cluster_analyses.append(result)

    logger.info(f"Completed analysis for {len(cluster_analyses)} clusters")

    return {"cluster_analyses": cluster_analyses}


async def _per_cluster_analysis_batched(
    final_clusters: list[ThematicCluster],
    paper_summaries: dict,
) -> dict[str, Any]:
    """Analyze clusters using Anthropic Batch API for 50% cost reduction."""
    processor = BatchProcessor(poll_interval=30)

    cluster_ids = []  # Track order for result mapping
    for i, cluster in enumerate(final_clusters):
        cluster_dois = cluster["paper_dois"]

        # Format papers for analysis
        papers_detail = []
        for doi in cluster_dois:
            summary = paper_summaries.get(doi)
            if not summary:
                continue

            paper_text = f"""
Title: {summary.get('title', 'Unknown')}
Year: {summary.get('year', 'Unknown')}
Summary: {summary.get('short_summary', 'N/A')}
Key Findings: {'; '.join(summary.get('key_findings', [])[:3])}
Methodology: {summary.get('methodology', 'N/A')[:150]}
Limitations: {'; '.join(summary.get('limitations', [])[:2])}"""
            papers_detail.append(paper_text)

        papers_text = "\n---\n".join(papers_detail)

        user_prompt = CLUSTER_ANALYSIS_USER_TEMPLATE.format(
            theme_name=cluster["label"],
            theme_description=cluster["description"],
            papers_detail=papers_text,
        )

        processor.add_request(
            custom_id=f"cluster-{i}",
            prompt=user_prompt,
            model=ModelTier.SONNET,
            max_tokens=4096,
            system=CLUSTER_ANALYSIS_SYSTEM_PROMPT,
        )
        cluster_ids.append(cluster["cluster_id"])

    logger.info(f"Submitting batch of {len(final_clusters)} clusters for analysis")
    results = await processor.execute_batch()

    cluster_analyses: list[ClusterAnalysis] = []
    for i, (cluster, cluster_id) in enumerate(zip(final_clusters, cluster_ids)):
        result = results.get(f"cluster-{i}")

        if result and result.success:
            try:
                content = result.content.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1])

                parsed = json.loads(content)

                cluster_analyses.append(ClusterAnalysis(
                    cluster_id=cluster_id,
                    narrative_summary=parsed.get("narrative_summary", ""),
                    timeline=parsed.get("timeline", []),
                    key_debates=parsed.get("key_debates", []),
                    methodologies=parsed.get("methodologies", []),
                    outstanding_questions=parsed.get("outstanding_questions", []),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse analysis for cluster {cluster['label']}: {e}")
                cluster_analyses.append(ClusterAnalysis(
                    cluster_id=cluster_id,
                    narrative_summary=f"Analysis parsing failed: {e}",
                    timeline=[],
                    key_debates=[],
                    methodologies=[],
                    outstanding_questions=[],
                ))
        else:
            error_msg = result.error if result else "No result returned"
            logger.warning(f"Cluster analysis failed for {cluster['label']}: {error_msg}")
            cluster_analyses.append(ClusterAnalysis(
                cluster_id=cluster_id,
                narrative_summary=f"Analysis failed: {error_msg}",
                timeline=[],
                key_debates=[],
                methodologies=[],
                outstanding_questions=[],
            ))

    logger.info(f"Completed analysis for {len(cluster_analyses)} clusters (batch)")

    return {"cluster_analyses": cluster_analyses}
