"""Per-cluster deep analysis implementation.

Uses unified invoke() for LLM calls.
"""

import logging
from typing import Any

from typing_extensions import TypedDict

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

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

    # Build batch prompts for all clusters
    prompts: list[str] = []
    cluster_ids: list[int] = []

    for cluster in final_clusters:
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

        prompts.append(user_prompt)
        cluster_ids.append(cluster["cluster_id"])

    logger.info(f"Submitting {len(final_clusters)} clusters for analysis")

    # Use invoke() for batch structured output
    try:
        results = await invoke(
            tier=ModelTier.SONNET,
            system=CLUSTER_ANALYSIS_SYSTEM_PROMPT,
            user=prompts,
            schema=ClusterAnalysisOutput,
            config=InvokeConfig(
                max_tokens=6144,
                batch_policy=BatchPolicy.PREFER_SPEED,
            ),
        )
    except Exception as e:
        logger.error(f"Cluster analysis batch failed: {e}")
        results = [None] * len(prompts)

    # Parse results
    cluster_analyses: list[ClusterAnalysis] = []
    for i, cluster in enumerate(final_clusters):
        cluster_id = cluster_ids[i]
        parsed = results[i] if results else None

        if parsed is not None:
            try:
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
            logger.warning(f"Cluster analysis failed for {cluster['label']}: No result returned")
            cluster_analyses.append(
                ClusterAnalysis(
                    cluster_id=cluster_id,
                    narrative_summary="Analysis failed: No result returned",
                    timeline=[],
                    key_debates=[],
                    methodologies=[],
                    outstanding_questions=[],
                )
            )

    logger.info(f"Completed analysis for {len(cluster_analyses)} clusters")

    return {"cluster_analyses": cluster_analyses}
