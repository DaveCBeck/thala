"""Cluster synthesis and conversion utilities."""

import json
import logging
from typing import Any

from workflows.research.academic_lit_review.state import (
    BERTopicCluster,
    LLMTopicSchema,
    PaperSummary,
    ThematicCluster,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

from .bertopic_clustering import _evaluate_bertopic_quality
from .prompts import OPUS_SYNTHESIS_SYSTEM_PROMPT, OPUS_SYNTHESIS_USER_TEMPLATE
from .schemas import OpusSynthesisOutput

logger = logging.getLogger(__name__)


async def synthesize_clusters_node(state: dict) -> dict[str, Any]:
    """Use Claude Opus to synthesize BERTopic and LLM clustering results.

    Opus reviews both approaches and decides on final clusters.
    Prefers LLM clustering when BERTopic produces poor results.
    """
    bertopic_clusters = state.get("bertopic_clusters", [])
    llm_schema = state.get("llm_topic_schema")
    paper_summaries = state.get("paper_summaries", {})
    total_papers = len(paper_summaries)

    # Handle case where only one clustering succeeded
    if not bertopic_clusters and not llm_schema:
        logger.error("Both clustering methods failed, cannot synthesize")
        return {
            "final_clusters": [],
            "cluster_labels": {},
        }

    # Evaluate BERTopic quality
    bertopic_good, bertopic_reason = _evaluate_bertopic_quality(
        bertopic_clusters, total_papers
    )

    if not bertopic_good:
        logger.warning(f"BERTopic quality issue: {bertopic_reason}")

    if not bertopic_clusters:
        # Use LLM clusters directly
        logger.info("Using LLM clusters only (BERTopic failed)")
        return _convert_llm_to_final_clusters(llm_schema, paper_summaries)

    if not llm_schema:
        # BERTopic only available
        if bertopic_good:
            logger.info("Using BERTopic clusters only (LLM failed)")
            return _convert_bertopic_to_final_clusters(bertopic_clusters, paper_summaries)
        else:
            # BERTopic is poor quality and LLM failed - return what we have with warning
            logger.warning(
                f"Using poor-quality BERTopic clusters (LLM failed). "
                f"Reason: {bertopic_reason}"
            )
            return _convert_bertopic_to_final_clusters(bertopic_clusters, paper_summaries)

    # Both succeeded - check if we should skip synthesis and use LLM directly
    if not bertopic_good:
        logger.info(
            f"Preferring LLM clusters over poor BERTopic results. "
            f"BERTopic issue: {bertopic_reason}"
        )
        return _convert_llm_to_final_clusters(llm_schema, paper_summaries)

    # Both succeeded - use Opus to synthesize
    try:
        # Format BERTopic results
        bertopic_summary = json.dumps(
            [
                {
                    "cluster_id": c["cluster_id"],
                    "topic_words": c["topic_words"],
                    "paper_count": len(c["paper_dois"]),
                    "paper_dois": c["paper_dois"][:10],  # Sample for brevity
                    "coherence": c["coherence_score"],
                }
                for c in bertopic_clusters
            ],
            indent=2,
        )

        # Format LLM results
        llm_summary = json.dumps(
            {
                "themes": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "paper_count": len(t["paper_dois"]),
                        "paper_dois": t["paper_dois"][:10],
                        "sub_themes": t["sub_themes"],
                    }
                    for t in llm_schema["themes"]
                ],
                "reasoning": llm_schema["reasoning"],
            },
            indent=2,
        )

        # Format paper summaries (brief version)
        papers_brief = json.dumps(
            {
                doi: {
                    "title": s.get("title", ""),
                    "year": s.get("year", 0),
                    "key_findings": s.get("key_findings", [])[:2],
                }
                for doi, s in list(paper_summaries.items())[:50]  # Sample for context
            },
            indent=2,
        )

        user_prompt = OPUS_SYNTHESIS_USER_TEMPLATE.format(
            paper_count=len(paper_summaries),
            bertopic_summary=bertopic_summary,
            llm_schema_summary=llm_summary,
            paper_summaries=papers_brief,
        )

        result: OpusSynthesisOutput = await get_structured_output(
            output_schema=OpusSynthesisOutput,
            user_prompt=user_prompt,
            system_prompt=OPUS_SYNTHESIS_SYSTEM_PROMPT,
            tier=ModelTier.OPUS,
            max_tokens=16000,
        )

        # Convert Pydantic models to TypedDict for state compatibility
        final_clusters: list[ThematicCluster] = []
        cluster_labels: dict[str, int] = {}

        for cluster_data in result.final_clusters:
            cluster = ThematicCluster(
                cluster_id=cluster_data.cluster_id,
                label=cluster_data.label,
                description=cluster_data.description,
                paper_dois=cluster_data.paper_dois,
                key_papers=cluster_data.key_papers,
                sub_themes=cluster_data.sub_themes,
                conflicts=cluster_data.conflicts,
                gaps=cluster_data.gaps,
                source=cluster_data.source,
            )
            final_clusters.append(cluster)

            # Build DOI -> cluster mapping
            for doi in cluster["paper_dois"]:
                cluster_labels[doi] = cluster["cluster_id"]

        logger.info(
            f"Opus synthesis complete: {len(final_clusters)} final clusters. "
            f"Reasoning: {result.reasoning[:100]}..."
        )

        return {
            "final_clusters": final_clusters,
            "cluster_labels": cluster_labels,
        }

    except Exception as e:
        logger.error(f"Opus synthesis failed: {e}, falling back to LLM clusters")
        return _convert_llm_to_final_clusters(llm_schema, paper_summaries)


def _convert_llm_to_final_clusters(
    llm_schema: LLMTopicSchema,
    paper_summaries: dict[str, PaperSummary],
) -> dict[str, Any]:
    """Convert LLM theme schema to final ThematicClusters."""
    final_clusters: list[ThematicCluster] = []
    cluster_labels: dict[str, int] = {}

    for i, theme in enumerate(llm_schema["themes"]):
        # Identify key papers (most cited in the cluster)
        cluster_dois = theme["paper_dois"]
        key_papers = _identify_key_papers(cluster_dois, paper_summaries)

        cluster = ThematicCluster(
            cluster_id=i,
            label=theme["name"],
            description=theme["description"],
            paper_dois=cluster_dois,
            key_papers=key_papers,
            sub_themes=theme["sub_themes"],
            conflicts=[],  # Will be filled by per-cluster analysis
            gaps=[],
            source="llm",
        )
        final_clusters.append(cluster)

        for doi in cluster_dois:
            cluster_labels[doi] = i

    return {
        "final_clusters": final_clusters,
        "cluster_labels": cluster_labels,
    }


def _convert_bertopic_to_final_clusters(
    bertopic_clusters: list[BERTopicCluster],
    paper_summaries: dict[str, PaperSummary],
) -> dict[str, Any]:
    """Convert BERTopic clusters to final ThematicClusters."""
    final_clusters: list[ThematicCluster] = []
    cluster_labels: dict[str, int] = {}

    for i, btc in enumerate(bertopic_clusters):
        cluster_dois = btc["paper_dois"]
        key_papers = _identify_key_papers(cluster_dois, paper_summaries)

        # Generate label from topic words
        label = " & ".join(btc["topic_words"][:3]).title() if btc["topic_words"] else f"Cluster {i}"

        cluster = ThematicCluster(
            cluster_id=i,
            label=label,
            description=f"Statistical cluster based on: {', '.join(btc['topic_words'][:5])}",
            paper_dois=cluster_dois,
            key_papers=key_papers,
            sub_themes=btc["topic_words"][3:8],  # Use remaining words as sub-themes
            conflicts=[],
            gaps=[],
            source="bertopic",
        )
        final_clusters.append(cluster)

        for doi in cluster_dois:
            cluster_labels[doi] = i

    return {
        "final_clusters": final_clusters,
        "cluster_labels": cluster_labels,
    }


def _identify_key_papers(
    dois: list[str],
    paper_summaries: dict[str, PaperSummary],
    max_key: int = 5,
) -> list[str]:
    """Identify key papers in a cluster by citation count and recency."""
    papers_with_scores = []

    for doi in dois:
        summary = paper_summaries.get(doi)
        if not summary:
            continue

        # Score based on relevance score and year
        relevance = summary.get("relevance_score", 0.5)
        year = summary.get("year", 2000)

        # Composite score: higher relevance and more recent = higher score
        score = relevance * 0.7 + (year - 1990) / 40 * 0.3

        papers_with_scores.append((doi, score))

    # Sort by score descending
    papers_with_scores.sort(key=lambda x: x[1], reverse=True)

    return [doi for doi, _ in papers_with_scores[:max_key]]
