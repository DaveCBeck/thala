"""Dual-strategy clustering subgraph for thematic organization of papers.

Implements parallel clustering approaches:
1. BERTopic statistical clustering (embedding + HDBSCAN)
2. LLM semantic clustering (Sonnet 4.5 with 1M context)
3. Opus synthesis to merge both approaches into final ThematicClusters

Flow:
    START -> prepare_documents -> [parallel: bertopic_clustering, llm_clustering]
          -> synthesize_clusters -> per_cluster_analysis -> END
"""

import asyncio
import json
import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from workflows.research.subgraphs.academic_lit_review.state import (
    BERTopicCluster,
    LitReviewInput,
    LLMTheme,
    LLMTopicSchema,
    PaperSummary,
    QualitySettings,
    ThematicCluster,
)
from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    invoke_with_cache,
)

logger = logging.getLogger(__name__)

# Constants
MIN_CLUSTER_SIZE = 3  # Minimum papers per cluster
MAX_CLUSTERS = 15  # Maximum number of final clusters
MIN_CLUSTERS = 5  # Minimum number of final clusters


# =============================================================================
# State Definition
# =============================================================================


class ClusterAnalysis(TypedDict):
    """Deep analysis of a single thematic cluster."""

    cluster_id: int
    narrative_summary: str  # 2-3 paragraph summary of the theme
    timeline: list[str]  # Key developments chronologically
    key_debates: list[str]  # Main debates and positions
    methodologies: list[str]  # Common methodological approaches
    outstanding_questions: list[str]  # Open research questions


class ClusteringState(TypedDict):
    """State for dual-strategy thematic clustering subgraph."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings
    paper_summaries: dict[str, PaperSummary]

    # Prepared documents for clustering
    document_texts: list[str]  # Formatted texts for BERTopic
    document_dois: list[str]  # DOIs in same order as document_texts

    # Parallel clustering results
    bertopic_clusters: Optional[list[BERTopicCluster]]
    bertopic_error: Optional[str]
    llm_topic_schema: Optional[LLMTopicSchema]
    llm_error: Optional[str]

    # Synthesized result
    final_clusters: list[ThematicCluster]
    cluster_labels: dict[str, int]  # DOI -> cluster_id

    # Per-cluster analysis
    cluster_analyses: list[ClusterAnalysis]


class FinalClusteringDecision(TypedDict):
    """Output schema for Opus cluster synthesis."""

    reasoning: str  # Explanation of synthesis decisions
    final_clusters: list[ThematicCluster]


# =============================================================================
# Prompts
# =============================================================================

LLM_CLUSTERING_SYSTEM_PROMPT = """You are an expert academic researcher analyzing a corpus of papers to identify coherent thematic clusters for a literature review.

Your task is to organize papers into 5-15 themes that would serve as natural sections in a literature review. Consider:

1. **Research Topics & Questions**: What fundamental questions do papers address?
2. **Methodological Approaches**: Are there distinct methodological camps?
3. **Theoretical Frameworks**: What theoretical lenses are used?
4. **Application Domains**: Are papers applied in specific contexts?
5. **Temporal Developments**: Are there clear phases of development?

For each theme:
- Choose a clear, descriptive name (suitable as a section heading)
- Write a 2-3 sentence description
- List the DOIs of papers belonging to this theme
- Note any sub-themes if the cluster is broad
- Describe how this theme relates to other themes

Papers may belong to multiple themes if they genuinely bridge topics.

Output a JSON object with this structure:
{
  "themes": [
    {
      "name": "Theme Name",
      "description": "What this theme covers...",
      "paper_dois": ["doi1", "doi2", ...],
      "sub_themes": ["sub1", "sub2"],
      "relationships": ["relates to Theme X because..."]
    }
  ],
  "reasoning": "Overall explanation of the thematic structure..."
}"""

LLM_CLUSTERING_USER_TEMPLATE = """I have {paper_count} academic papers to organize into thematic clusters for a literature review.

Research Topic: {topic}

Research Questions:
{research_questions}

Here are all the paper summaries:

{summaries}

Analyze these papers and organize them into coherent thematic clusters."""

OPUS_SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing two different clustering analyses of academic papers:
1. Statistical clustering (BERTopic) - based on text embeddings and density
2. Semantic clustering (LLM analysis) - based on conceptual understanding

Your task is to create the final thematic organization by:
1. Comparing where the two approaches agree and disagree
2. Deciding which clusters to keep, merge, or split
3. Assigning final theme names and descriptions
4. Identifying key papers and research gaps per cluster

Guidelines:
- Prefer semantic coherence over statistical purity
- Aim for 5-15 final clusters (suitable as review sections)
- Ensure every paper is assigned to at least one cluster
- Mark papers that bridge multiple themes
- Note conflicts and gaps within each theme

Output a JSON object with:
{
  "reasoning": "Explanation of your synthesis decisions...",
  "final_clusters": [
    {
      "cluster_id": 0,
      "label": "Theme Name",
      "description": "What this cluster covers...",
      "paper_dois": ["doi1", "doi2"],
      "key_papers": ["most_important_doi1", "most_important_doi2"],
      "sub_themes": ["sub1", "sub2"],
      "conflicts": ["papers X and Y disagree on..."],
      "gaps": ["under-researched area..."],
      "source": "merged" | "bertopic" | "llm"
    }
  ]
}"""

OPUS_SYNTHESIS_USER_TEMPLATE = """I have {paper_count} papers to organize. Here are the two clustering analyses:

## Statistical Clustering (BERTopic)
{bertopic_summary}

## Semantic Clustering (LLM Analysis)
{llm_schema_summary}

## Paper Summaries (for reference)
{paper_summaries}

Synthesize these into a final thematic organization."""

CLUSTER_ANALYSIS_SYSTEM_PROMPT = """You are a research analyst providing deep analysis of a thematic cluster of papers.

For this cluster, produce:
1. **Narrative Summary**: A 2-3 paragraph overview of the theme, suitable as the introduction to a literature review section
2. **Timeline**: Key developments in chronological order
3. **Key Debates**: Main disagreements, tensions, or alternative positions
4. **Methodologies**: Common methodological approaches used
5. **Outstanding Questions**: Open research questions and gaps

Be specific and cite papers by their titles when making claims.

Output a JSON object:
{
  "narrative_summary": "...",
  "timeline": ["1995: Paper X established...", "2010: Paper Y challenged..."],
  "key_debates": ["Whether approach A or B is superior...", "The role of factor C..."],
  "methodologies": ["Survey studies (Paper A, Paper B)", "Experiments (Paper C)"],
  "outstanding_questions": ["How does X affect Y?", "Can findings generalize to..."]
}"""

CLUSTER_ANALYSIS_USER_TEMPLATE = """Analyze this thematic cluster:

Theme: {theme_name}
Description: {theme_description}

Papers in this cluster:
{papers_detail}

Provide a deep analysis of this research theme."""


# =============================================================================
# Helper Functions
# =============================================================================


def prepare_document_for_clustering(summary: PaperSummary) -> str:
    """Format a paper summary for BERTopic clustering.

    Creates a text representation combining title, abstract, key findings,
    and themes for embedding-based clustering.
    """
    parts = [
        summary.get("title", "Untitled"),
    ]

    if summary.get("short_summary"):
        parts.append(summary["short_summary"])

    key_findings = summary.get("key_findings", [])
    if key_findings:
        parts.append("Key findings: " + "; ".join(key_findings[:5]))

    themes = summary.get("themes", [])
    if themes:
        parts.append("Themes: " + ", ".join(themes[:10]))

    methodology = summary.get("methodology")
    if methodology:
        parts.append(f"Methodology: {methodology}")

    return "\n".join(parts)


def format_paper_for_llm(doi: str, summary: PaperSummary) -> str:
    """Format a paper summary for LLM clustering prompt."""
    authors = summary.get("authors", [])
    authors_str = ", ".join(authors[:3])
    if len(authors) > 3:
        authors_str += " et al."

    key_findings = summary.get("key_findings", [])
    findings_str = "; ".join(key_findings[:3]) if key_findings else "Not extracted"

    return f"""DOI: {doi}
Title: {summary.get('title', 'Untitled')}
Authors: {authors_str}
Year: {summary.get('year', 'Unknown')}
Venue: {summary.get('venue', 'Unknown')}
Summary: {summary.get('short_summary', 'No summary available')[:500]}
Key Findings: {findings_str}
Methodology: {summary.get('methodology', 'Not specified')[:200]}
Themes: {', '.join(summary.get('themes', [])[:5])}"""


# =============================================================================
# Node Functions
# =============================================================================


async def prepare_documents_node(state: ClusteringState) -> dict[str, Any]:
    """Prepare paper summaries as documents for clustering."""
    paper_summaries = state.get("paper_summaries", {})

    if not paper_summaries:
        logger.warning("No paper summaries to cluster")
        return {
            "document_texts": [],
            "document_dois": [],
        }

    document_texts = []
    document_dois = []

    for doi, summary in paper_summaries.items():
        doc_text = prepare_document_for_clustering(summary)
        document_texts.append(doc_text)
        document_dois.append(doi)

    logger.info(f"Prepared {len(document_texts)} documents for clustering")

    return {
        "document_texts": document_texts,
        "document_dois": document_dois,
    }


async def run_bertopic_clustering_node(state: ClusteringState) -> dict[str, Any]:
    """Statistical clustering using BERTopic.

    Process:
    1. Create document representations from paper summaries
    2. Embed documents using sentence-transformers
    3. Reduce dimensionality with UMAP
    4. Cluster with HDBSCAN
    5. Extract topic representations
    """
    document_texts = state.get("document_texts", [])
    document_dois = state.get("document_dois", [])

    if len(document_texts) < MIN_CLUSTER_SIZE:
        logger.warning(
            f"Too few documents for BERTopic clustering: {len(document_texts)}"
        )
        return {
            "bertopic_clusters": [],
            "bertopic_error": "Too few documents for statistical clustering",
        }

    try:
        from bertopic import BERTopic

        # Configure BERTopic
        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=MIN_CLUSTER_SIZE,
            nr_topics="auto",  # Let it determine optimal number
            calculate_probabilities=True,
            verbose=False,
        )

        # Fit and transform
        topics, probs = topic_model.fit_transform(document_texts)

        # Build cluster output
        clusters: list[BERTopicCluster] = []
        topic_ids = set(topics)

        for topic_id in topic_ids:
            if topic_id == -1:  # Skip outliers
                continue

            # Get papers in this cluster
            cluster_dois = [
                document_dois[i] for i, t in enumerate(topics) if t == topic_id
            ]

            if not cluster_dois:
                continue

            # Get topic representation (top words)
            topic_info = topic_model.get_topic(topic_id)
            topic_words = [word for word, _ in topic_info[:10]] if topic_info else []

            # Calculate average probability for coherence score
            cluster_indices = [i for i, t in enumerate(topics) if t == topic_id]
            coherence = float(probs[cluster_indices].mean()) if len(cluster_indices) > 0 else 0.0

            clusters.append(
                BERTopicCluster(
                    cluster_id=int(topic_id),
                    topic_words=topic_words,
                    paper_dois=cluster_dois,
                    coherence_score=coherence,
                )
            )

        # Handle outlier papers (topic_id == -1)
        outlier_dois = [
            document_dois[i] for i, t in enumerate(topics) if t == -1
        ]
        if outlier_dois:
            logger.info(f"BERTopic: {len(outlier_dois)} papers not assigned to clusters")

        logger.info(
            f"BERTopic clustering complete: {len(clusters)} clusters "
            f"from {len(document_texts)} documents"
        )

        return {
            "bertopic_clusters": clusters,
            "bertopic_error": None,
        }

    except ImportError:
        error_msg = "BERTopic not installed. Install with: pip install bertopic"
        logger.error(error_msg)
        return {
            "bertopic_clusters": [],
            "bertopic_error": error_msg,
        }
    except Exception as e:
        error_msg = f"BERTopic clustering failed: {str(e)}"
        logger.error(error_msg)
        return {
            "bertopic_clusters": [],
            "bertopic_error": error_msg,
        }


async def run_llm_clustering_node(state: ClusteringState) -> dict[str, Any]:
    """Semantic clustering using Claude Sonnet 4.5 with 1M context.

    Feeds ALL paper summaries to a single LLM call, leveraging the
    1M token context window to see the entire corpus at once.
    """
    paper_summaries = state.get("paper_summaries", {})
    input_data = state.get("input", {})

    if not paper_summaries:
        logger.warning("No paper summaries for LLM clustering")
        return {
            "llm_topic_schema": None,
            "llm_error": "No paper summaries available",
        }

    topic = input_data.get("topic", "Unknown topic")
    research_questions = input_data.get("research_questions", [])

    # Format all summaries for the prompt
    summaries_text = "\n\n---\n\n".join(
        format_paper_for_llm(doi, summary)
        for doi, summary in paper_summaries.items()
    )

    # Format research questions
    rq_text = "\n".join(f"- {q}" for q in research_questions) if research_questions else "None specified"

    user_prompt = LLM_CLUSTERING_USER_TEMPLATE.format(
        paper_count=len(paper_summaries),
        topic=topic,
        research_questions=rq_text,
        summaries=summaries_text,
    )

    try:
        # Use Sonnet with high max_tokens for 1M context handling
        llm = get_llm(
            tier=ModelTier.SONNET,
            max_tokens=16000,
        )

        response = await invoke_with_cache(
            llm,
            system_prompt=LLM_CLUSTERING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cache_ttl="1h",  # Use extended TTL for large prompts
        )

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        content = content.strip()

        # Parse JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)

        # Convert to LLMTopicSchema
        themes: list[LLMTheme] = []
        for theme_data in result.get("themes", []):
            themes.append(
                LLMTheme(
                    name=theme_data.get("name", "Unnamed Theme"),
                    description=theme_data.get("description", ""),
                    paper_dois=theme_data.get("paper_dois", []),
                    sub_themes=theme_data.get("sub_themes", []),
                    relationships=theme_data.get("relationships", []),
                )
            )

        llm_schema = LLMTopicSchema(
            themes=themes,
            reasoning=result.get("reasoning", ""),
        )

        logger.info(
            f"LLM clustering complete: {len(themes)} themes "
            f"from {len(paper_summaries)} papers"
        )

        return {
            "llm_topic_schema": llm_schema,
            "llm_error": None,
        }

    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM clustering response: {e}"
        logger.error(error_msg)
        return {
            "llm_topic_schema": None,
            "llm_error": error_msg,
        }
    except Exception as e:
        error_msg = f"LLM clustering failed: {str(e)}"
        logger.error(error_msg)
        return {
            "llm_topic_schema": None,
            "llm_error": error_msg,
        }


async def synthesize_clusters_node(state: ClusteringState) -> dict[str, Any]:
    """Use Claude Opus to synthesize BERTopic and LLM clustering results.

    Opus reviews both approaches and decides on final clusters.
    """
    bertopic_clusters = state.get("bertopic_clusters", [])
    llm_schema = state.get("llm_topic_schema")
    paper_summaries = state.get("paper_summaries", {})

    # Handle case where only one clustering succeeded
    if not bertopic_clusters and not llm_schema:
        logger.error("Both clustering methods failed, cannot synthesize")
        return {
            "final_clusters": [],
            "cluster_labels": {},
        }

    if not bertopic_clusters:
        # Use LLM clusters directly
        logger.info("Using LLM clusters only (BERTopic failed)")
        return _convert_llm_to_final_clusters(llm_schema, paper_summaries)

    if not llm_schema:
        # Use BERTopic clusters directly
        logger.info("Using BERTopic clusters only (LLM failed)")
        return _convert_bertopic_to_final_clusters(bertopic_clusters, paper_summaries)

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

        llm = get_llm(tier=ModelTier.OPUS, max_tokens=16000)

        response = await invoke_with_cache(
            llm,
            system_prompt=OPUS_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cache_ttl="1h",
        )

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        content = content.strip()

        # Parse JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)

        # Convert to ThematicCluster list
        final_clusters: list[ThematicCluster] = []
        cluster_labels: dict[str, int] = {}

        for cluster_data in result.get("final_clusters", []):
            cluster = ThematicCluster(
                cluster_id=cluster_data.get("cluster_id", len(final_clusters)),
                label=cluster_data.get("label", "Unnamed"),
                description=cluster_data.get("description", ""),
                paper_dois=cluster_data.get("paper_dois", []),
                key_papers=cluster_data.get("key_papers", []),
                sub_themes=cluster_data.get("sub_themes", []),
                conflicts=cluster_data.get("conflicts", []),
                gaps=cluster_data.get("gaps", []),
                source=cluster_data.get("source", "merged"),
            )
            final_clusters.append(cluster)

            # Build DOI -> cluster mapping
            for doi in cluster["paper_dois"]:
                cluster_labels[doi] = cluster["cluster_id"]

        logger.info(
            f"Opus synthesis complete: {len(final_clusters)} final clusters. "
            f"Reasoning: {result.get('reasoning', 'N/A')[:100]}..."
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


async def per_cluster_analysis_node(state: ClusteringState) -> dict[str, Any]:
    """Generate deep analysis for each thematic cluster."""
    final_clusters = state.get("final_clusters", [])
    paper_summaries = state.get("paper_summaries", {})

    if not final_clusters:
        logger.warning("No clusters to analyze")
        return {"cluster_analyses": []}

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

            response = await invoke_with_cache(
                llm,
                system_prompt=CLUSTER_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
            content = content.strip()

            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            result = json.loads(content)

            return ClusterAnalysis(
                cluster_id=cluster["cluster_id"],
                narrative_summary=result.get("narrative_summary", ""),
                timeline=result.get("timeline", []),
                key_debates=result.get("key_debates", []),
                methodologies=result.get("methodologies", []),
                outstanding_questions=result.get("outstanding_questions", []),
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

    # Analyze clusters concurrently (with limit)
    semaphore = asyncio.Semaphore(5)

    async def analyze_with_limit(cluster: ThematicCluster) -> ClusterAnalysis:
        async with semaphore:
            return await analyze_single_cluster(cluster)

    tasks = [analyze_with_limit(c) for c in final_clusters]
    analyses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful analyses
    cluster_analyses: list[ClusterAnalysis] = []
    for result in analyses:
        if isinstance(result, Exception):
            logger.error(f"Cluster analysis task failed: {result}")
            continue
        cluster_analyses.append(result)

    logger.info(f"Completed analysis for {len(cluster_analyses)} clusters")

    return {"cluster_analyses": cluster_analyses}


# =============================================================================
# Subgraph Definition
# =============================================================================


def create_clustering_subgraph() -> StateGraph:
    """Create the dual-strategy clustering subgraph.

    Flow:
        START -> prepare_documents -> parallel[bertopic, llm]
              -> synthesize_clusters -> per_cluster_analysis -> END

    Note: LangGraph doesn't support true parallel execution, so we
    run clustering methods sequentially but could be parallelized
    with asyncio in a custom node.
    """
    builder = StateGraph(ClusteringState)

    # Add nodes
    builder.add_node("prepare_documents", prepare_documents_node)
    builder.add_node("bertopic_clustering", run_bertopic_clustering_node)
    builder.add_node("llm_clustering", run_llm_clustering_node)
    builder.add_node("synthesize_clusters", synthesize_clusters_node)
    builder.add_node("per_cluster_analysis", per_cluster_analysis_node)

    # Add edges
    # Sequential for now - could be parallelized with a custom parallel node
    builder.add_edge(START, "prepare_documents")
    builder.add_edge("prepare_documents", "bertopic_clustering")
    builder.add_edge("bertopic_clustering", "llm_clustering")
    builder.add_edge("llm_clustering", "synthesize_clusters")
    builder.add_edge("synthesize_clusters", "per_cluster_analysis")
    builder.add_edge("per_cluster_analysis", END)

    return builder.compile()


# Alternative with true parallel clustering
def create_parallel_clustering_subgraph() -> StateGraph:
    """Create clustering subgraph with parallel BERTopic and LLM clustering.

    Uses a custom node to run both clustering methods concurrently.
    """
    async def parallel_clustering_node(state: ClusteringState) -> dict[str, Any]:
        """Run BERTopic and LLM clustering in parallel."""
        bertopic_task = asyncio.create_task(run_bertopic_clustering_node(state))
        llm_task = asyncio.create_task(run_llm_clustering_node(state))

        bertopic_result, llm_result = await asyncio.gather(
            bertopic_task, llm_task, return_exceptions=True
        )

        result = {}

        if isinstance(bertopic_result, Exception):
            logger.error(f"BERTopic clustering failed: {bertopic_result}")
            result["bertopic_clusters"] = []
            result["bertopic_error"] = str(bertopic_result)
        else:
            result.update(bertopic_result)

        if isinstance(llm_result, Exception):
            logger.error(f"LLM clustering failed: {llm_result}")
            result["llm_topic_schema"] = None
            result["llm_error"] = str(llm_result)
        else:
            result.update(llm_result)

        return result

    builder = StateGraph(ClusteringState)

    builder.add_node("prepare_documents", prepare_documents_node)
    builder.add_node("parallel_clustering", parallel_clustering_node)
    builder.add_node("synthesize_clusters", synthesize_clusters_node)
    builder.add_node("per_cluster_analysis", per_cluster_analysis_node)

    builder.add_edge(START, "prepare_documents")
    builder.add_edge("prepare_documents", "parallel_clustering")
    builder.add_edge("parallel_clustering", "synthesize_clusters")
    builder.add_edge("synthesize_clusters", "per_cluster_analysis")
    builder.add_edge("per_cluster_analysis", END)

    return builder.compile()


# Export the parallel version as default
clustering_subgraph = create_parallel_clustering_subgraph()


# =============================================================================
# Convenience Function
# =============================================================================


async def run_clustering(
    paper_summaries: dict[str, PaperSummary],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
) -> dict[str, Any]:
    """Run dual-strategy clustering as a standalone operation.

    Args:
        paper_summaries: DOI -> PaperSummary mapping
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings

    Returns:
        Dict with final_clusters, cluster_labels, cluster_analyses,
        and intermediate results (bertopic_clusters, llm_topic_schema)
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=None,
        include_books=False,
        focus_areas=None,
        exclude_terms=None,
        max_papers=None,
    )

    initial_state = ClusteringState(
        input=input_data,
        quality_settings=quality_settings,
        paper_summaries=paper_summaries,
        document_texts=[],
        document_dois=[],
        bertopic_clusters=None,
        bertopic_error=None,
        llm_topic_schema=None,
        llm_error=None,
        final_clusters=[],
        cluster_labels={},
        cluster_analyses=[],
    )

    result = await clustering_subgraph.ainvoke(initial_state)

    return {
        "final_clusters": result.get("final_clusters", []),
        "cluster_labels": result.get("cluster_labels", {}),
        "cluster_analyses": result.get("cluster_analyses", []),
        "bertopic_clusters": result.get("bertopic_clusters", []),
        "llm_topic_schema": result.get("llm_topic_schema"),
        "bertopic_error": result.get("bertopic_error"),
        "llm_error": result.get("llm_error"),
    }
