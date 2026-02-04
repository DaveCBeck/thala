"""Per-cluster deep analysis implementation.

Routes through central LLM broker for unified cost/speed management.
"""

import logging
from typing import Any

from typing_extensions import TypedDict

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import ModelTier
from workflows.shared.llm_utils.structured import (
    StructuredRequest,
    get_structured_output,
)

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

    Routes through central LLM broker for unified cost/speed management.
    """
    final_clusters = state.get("final_clusters", [])
    paper_summaries = state.get("paper_summaries", {})

    if not final_clusters:
        logger.warning("No clusters to analyze")
        return {"cluster_analyses": []}

    # Build batch requests for all clusters
    requests = []
    cluster_id_map = {}

    for i, cluster in enumerate(final_clusters):
        cluster_dois = cluster["paper_dois"]

        # Format papers for analysis
        papers_detail = []
        for doi in cluster_dois:
            summary = paper_summaries.get(doi)
            if not summary:
                continue

            paper_text = f"""
Title: {summary.get("title", "Unknown")}
Year: {summary.get("year", "Unknown")}
Summary: {summary.get("short_summary", "N/A")}
Key Findings: {"; ".join(summary.get("key_findings", [])[:3])}
Methodology: {summary.get("methodology", "N/A")[:150]}
Limitations: {"; ".join(summary.get("limitations", [])[:2])}"""
            papers_detail.append(paper_text)

        papers_text = "\n---\n".join(papers_detail)

        user_prompt = CLUSTER_ANALYSIS_USER_TEMPLATE.format(
            theme_name=cluster["label"],
            theme_description=cluster["description"],
            papers_detail=papers_text,
        )

        request_id = f"cluster-{i}"
        requests.append(
            StructuredRequest(
                id=request_id,
                user_prompt=user_prompt,
            )
        )
        cluster_id_map[request_id] = cluster["cluster_id"]

    logger.info(f"Submitting {len(final_clusters)} clusters for analysis via broker")

    # Route through broker with PREFER_BALANCE policy
    batch_results = await get_structured_output(
        output_schema=ClusterAnalysisOutput,
        requests=requests,
        system_prompt=CLUSTER_ANALYSIS_SYSTEM_PROMPT,
        tier=ModelTier.SONNET,
        max_tokens=4096,
        batch_policy=BatchPolicy.PREFER_BALANCE,
    )

    # Parse results
    cluster_analyses: list[ClusterAnalysis] = []
    for i, cluster in enumerate(final_clusters):
        request_id = f"cluster-{i}"
        cluster_id = cluster_id_map[request_id]
        result = batch_results.results.get(request_id)

        if result and result.success:
            try:
                parsed = result.value

                cluster_analyses.append(
                    ClusterAnalysis(
                        cluster_id=cluster_id,
                        narrative_summary=parsed.narrative_summary,
                        timeline=parsed.timeline,
                        key_debates=parsed.key_debates,
                        methodologies=parsed.methodologies,
                        outstanding_questions=parsed.outstanding_questions,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse analysis for cluster {cluster['label']}: {e}")
                cluster_analyses.append(
                    ClusterAnalysis(
                        cluster_id=cluster_id,
                        narrative_summary=f"Analysis parsing failed: {e}",
                        timeline=[],
                        key_debates=[],
                        methodologies=[],
                        outstanding_questions=[],
                    )
                )
        else:
            error_msg = result.error if result else "No result returned"
            logger.warning(f"Cluster analysis failed for {cluster['label']}: {error_msg}")
            cluster_analyses.append(
                ClusterAnalysis(
                    cluster_id=cluster_id,
                    narrative_summary=f"Analysis failed: {error_msg}",
                    timeline=[],
                    key_debates=[],
                    methodologies=[],
                    outstanding_questions=[],
                )
            )

    logger.info(f"Completed analysis for {len(cluster_analyses)} clusters")

    return {"cluster_analyses": cluster_analyses}
