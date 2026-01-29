---
name: citation-network-academic-review-workflow
title: "Citation Network Academic Review Workflow"
date: 2025-12-30
category: langgraph
applicability:
  - "When generating comprehensive academic literature reviews from scratch"
  - "When citation network expansion is needed to discover related papers"
  - "When dual-strategy clustering (statistical + semantic) improves thematic analysis"
  - "When quality-tiered output is required (quick→high_quality presets)"
components: [langgraph_workflow, citation_graph, clustering_pipeline, openalex_api]
complexity: complex
verified_in_production: true
related_solutions: []
tags: [langgraph, citation-network, networkx, bertopic, openalex, diffusion-algorithm, thematic-clustering, literature-review, quality-presets]
shared: true
gist_url: https://gist.github.com/DaveCBeck/7d688d955f8ba6c15146e62a77408e9d
article_path: .context/libs/thala-dev/content/2025-12-30-citation-network-academic-review-workflow-langgraph.md
---

# Citation Network Academic Review Workflow

## Intent

Generate PhD-equivalent academic literature reviews (10-25k words, 50-300+ sources) through a 5-phase LangGraph pipeline: discovery (keyword search + citation seeding), diffusion (recursive citation expansion with two-stage relevance filtering), processing (full-text acquisition and summarization), clustering (dual BERTopic + LLM semantic analysis), and synthesis (parallel section writing with PRISMA verification).

## Motivation

Academic literature reviews require:

1. **Comprehensive discovery**: Beyond keyword search—citation networks reveal foundational and recent works
2. **Quality filtering**: Raw search returns noise; relevance filtering is essential
3. **Thematic organization**: Papers must be grouped into coherent themes for synthesis
4. **Structured output**: Reviews need consistent sections, citations, and methodology documentation

This pattern addresses all four through a modular LangGraph workflow with quality presets that scale from quick (50 papers, 8k words) to high_quality (300 papers, 25k words).

## Applicability

Use this pattern when:
- Generating systematic literature reviews for academic or professional research
- Citation network analysis is valuable for discovery (papers reference each other)
- Output quality must scale with available time/budget (quality presets)
- Thematic clustering is needed to organize large paper corpora

Do NOT use this pattern when:
- Simple keyword search suffices (no citation expansion needed)
- Corpus is already curated (skip discovery/diffusion phases)
- Output is informal summaries (no PRISMA/methodology documentation needed)
- Time constraints prohibit multi-hour processing

## Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Academic Literature Review Workflow                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   START                                                                     │
│     │                                                                       │
│     ▼                                                                       │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║ Phase 1: DISCOVERY                                                 ║    │
│   ║ • LLM-generated OpenAlex queries                                   ║    │
│   ║ • Forward/backward citation seeding                                ║    │
│   ║ • Expert author identification                                     ║    │
│   ║ • Output: initial paper corpus (DOIs)                              ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│     │                                                                       │
│     ▼                                                                       │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║ Phase 2: DIFFUSION                                                 ║    │
│   ║ • Recursive citation expansion (2-5 stages)                        ║    │
│   ║ • Two-stage filtering: co-citation → LLM scoring                   ║    │
│   ║ • Saturation detection (coverage delta threshold)                  ║    │
│   ║ • Output: expanded corpus + citation graph                         ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│     │                                                                       │
│     ▼                                                                       │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║ Phase 3: PROCESSING                                                ║    │
│   ║ • Full-text acquisition via retrieve-academic                      ║    │
│   ║ • Document processing (Marker PDF extraction)                      ║    │
│   ║ • PaperSummary extraction (findings, methodology, claims)          ║    │
│   ║ • Output: paper_summaries dict with structured data                ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│     │                                                                       │
│     ▼                                                                       │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║ Phase 4: CLUSTERING                                                ║    │
│   ║ • BERTopic statistical clustering (embeddings)                     ║    │
│   ║ • LLM semantic clustering (Sonnet 4.5 1M context)                  ║    │
│   ║ • Opus synthesis (merges both approaches)                          ║    │
│   ║ • Output: ThematicClusters with labels, conflicts, gaps            ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│     │                                                                       │
│     ▼                                                                       │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║ Phase 5: SYNTHESIS                                                 ║    │
│   ║ • Parallel thematic section drafting                               ║    │
│   ║ • Intro/methodology/discussion/conclusions writing                 ║    │
│   ║ • Citation processing to Pandoc format                             ║    │
│   ║ • PRISMA documentation generation                                  ║    │
│   ║ • Output: final_review (10-25k words)                              ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│     │                                                                       │
│     ▼                                                                       │
│   END                                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Define Quality Presets

Quality presets control workflow depth and output size:

```python
# workflows/research/academic_lit_review/quality_presets.py

from typing_extensions import TypedDict


class QualitySettings(TypedDict):
    """Configuration for a quality tier."""

    max_stages: int  # Maximum diffusion stages
    max_papers: int  # Maximum papers to process
    target_word_count: int  # Target length of final review
    min_citations_filter: int  # Minimum citations for discovery
    saturation_threshold: float  # Coverage delta threshold
    use_batch_api: bool  # Use Anthropic Batch API
    supervision_loops: str  # Which supervision loops to run
    recency_years: int  # Years to consider "recent"
    recency_quota: float  # Target fraction of recent papers


QUALITY_PRESETS: dict[str, QualitySettings] = {
    "quick": QualitySettings(
        max_stages=2,
        max_papers=50,
        target_word_count=8000,
        min_citations_filter=5,
        saturation_threshold=0.15,
        use_batch_api=True,
        supervision_loops="all",
        recency_years=3,
        recency_quota=0.25,
    ),
    "standard": QualitySettings(
        max_stages=3,
        max_papers=100,
        target_word_count=12000,
        min_citations_filter=10,
        saturation_threshold=0.12,
        use_batch_api=True,
        supervision_loops="all",
        recency_years=3,
        recency_quota=0.25,
    ),
    "comprehensive": QualitySettings(
        max_stages=4,
        max_papers=200,
        target_word_count=17500,
        min_citations_filter=10,
        saturation_threshold=0.10,
        use_batch_api=True,
        supervision_loops="all",
        recency_years=3,
        recency_quota=0.25,
    ),
    "high_quality": QualitySettings(
        max_stages=5,
        max_papers=300,
        target_word_count=25000,
        min_citations_filter=10,
        saturation_threshold=0.10,
        use_batch_api=True,
        supervision_loops="all",
        recency_years=3,
        recency_quota=0.25,
    ),
}
```

### Step 2: Define State with Reducers for Parallel Aggregation

Use `Annotated` with reducer functions for parallel node outputs:

```python
# workflows/research/academic_lit_review/state.py

from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from .reducers import merge_dicts, merge_paper_summaries
from .quality_presets import QualitySettings


class LitReviewInput(TypedDict):
    """Input parameters for literature review workflow."""

    topic: str
    research_questions: list[str]
    quality: Literal["test", "quick", "standard", "comprehensive", "high_quality"]
    date_range: Optional[tuple[int, int]]
    language_code: Optional[str]


class PaperMetadata(TypedDict):
    """Core metadata for a discovered paper."""

    doi: str
    title: str
    authors: list[dict]
    year: int
    venue: Optional[str]
    cited_by_count: int
    abstract: Optional[str]
    openalex_id: str
    discovery_stage: int  # Which diffusion stage found this
    discovery_method: str  # "keyword", "forward", "backward", "expert"
    relevance_score: Optional[float]


class AcademicLitReviewState(TypedDict):
    """Complete state for academic literature review workflow."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings

    # Discovery (parallel aggregation via `add` reducer)
    keyword_papers: Annotated[list[str], add]
    citation_papers: Annotated[list[str], add]
    expert_papers: Annotated[list[str], add]

    # Paper corpus (accumulated via merge_dicts reducer)
    paper_corpus: Annotated[dict[str, PaperMetadata], merge_dicts]
    paper_summaries: Annotated[dict[str, "PaperSummary"], merge_paper_summaries]
    citation_edges: Annotated[list["CitationEdge"], add]

    # Clustering results
    bertopic_clusters: Optional[list["BERTopicCluster"]]
    llm_topic_schema: Optional["LLMTopicSchema"]
    clusters: list["ThematicCluster"]

    # Synthesis outputs
    section_drafts: dict[str, str]
    final_review: Optional[str]
    references: list["FormattedCitation"]
    prisma_documentation: Optional[str]

    # Workflow metadata
    current_phase: str
    errors: Annotated[list[dict], add]
```

### Step 3: Build Citation Graph with NetworkX

Track citation relationships for analysis and co-citation filtering:

```python
# workflows/research/academic_lit_review/citation_graph/builder.py

import networkx as nx
from datetime import datetime
from typing import Optional

from ..state import CitationEdge, PaperMetadata, PaperNode


class CitationGraphBuilder:
    """Manages citation graph construction and analysis."""

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, PaperNode] = {}
        self._edges: list[CitationEdge] = []
        self._cached_centrality: Optional[dict[str, float]] = None

    def add_paper(self, doi: str, metadata: PaperMetadata) -> None:
        """Add or update paper node."""
        if doi not in self._nodes:
            self._nodes[doi] = PaperNode(
                doi=doi,
                title=metadata.get("title", ""),
                year=metadata.get("year", 0),
                cited_by_count=metadata.get("cited_by_count", 0),
                in_degree=0,
                out_degree=0,
                discovery_stage=metadata.get("discovery_stage", 0),
                cluster_id=None,
            )
            self._graph.add_node(doi)
        self.invalidate_cache()

    def add_citation(
        self, citing_doi: str, cited_doi: str, edge_type: str = "forward"
    ) -> bool:
        """Add directed edge (citing -> cited)."""
        if citing_doi not in self._nodes or cited_doi not in self._nodes:
            return False

        if self._graph.has_edge(citing_doi, cited_doi):
            return False

        self._graph.add_edge(citing_doi, cited_doi)
        self._edges.append(
            CitationEdge(
                citing_doi=citing_doi,
                cited_doi=cited_doi,
                edge_type=edge_type,
            )
        )

        # Update degrees
        self._nodes[citing_doi]["out_degree"] += 1
        self._nodes[cited_doi]["in_degree"] += 1

        self.invalidate_cache()
        return True

    def get_corpus_overlap_count(
        self, paper_doi: str, corpus_dois: set[str]
    ) -> int:
        """Count how many corpus papers cite or are cited by this paper."""
        if paper_doi not in self._graph:
            return 0

        # Papers this one cites that are in corpus
        citing = set(self._graph.successors(paper_doi)) & corpus_dois
        # Papers that cite this one that are in corpus
        cited_by = set(self._graph.predecessors(paper_doi)) & corpus_dois

        return len(citing | cited_by)

    def invalidate_cache(self) -> None:
        """Invalidate cached analysis results."""
        self._cached_centrality = None
```

### Step 4: Implement Two-Stage Relevance Filtering

Filter candidates using corpus co-citation counts, then LLM scoring:

```python
# workflows/research/academic_lit_review/diffusion_engine/relevance_filters.py

from typing import Any

from ..citation_graph import CitationGraph
from ..state import FallbackCandidate
from ..utils import batch_score_relevance
from workflows.shared.llm_utils import ModelTier


async def enrich_with_cocitation_counts_node(
    state: "DiffusionEngineState",
) -> dict[str, Any]:
    """Stage 1: Compute corpus co-citation counts for each candidate.

    Adds 'corpus_cocitations' field indicating how many corpus papers
    cite or are cited by this paper. Passed to LLM as context.
    """
    candidates = state.get("current_stage_candidates", [])
    citation_graph = state.get("citation_graph")
    corpus_dois = set(state.get("paper_corpus", {}).keys())

    if not candidates or not citation_graph:
        return {"current_stage_candidates": candidates}

    for candidate in candidates:
        doi = candidate.get("doi")
        if doi:
            candidate["corpus_cocitations"] = citation_graph.get_corpus_overlap_count(
                paper_doi=doi,
                corpus_dois=corpus_dois,
            )

    return {"current_stage_candidates": candidates}


async def score_relevance_node(state: "DiffusionEngineState") -> dict[str, Any]:
    """Stage 2: Score all candidates with LLM using co-citation context.

    Returns:
    - relevant papers (score >= 0.6)
    - fallback candidates (score 0.5-0.6) for the fallback queue
    - rejected papers (score < 0.5)
    """
    candidates = state.get("current_stage_candidates", [])
    topic = state["input"]["topic"]
    research_questions = state["input"].get("research_questions", [])

    if not candidates:
        return {
            "current_stage_relevant": [],
            "current_stage_rejected": [],
            "current_stage_fallback": [],
        }

    # LLM scoring returns 3-tuple: (relevant, fallback, rejected)
    relevant, fallback_candidates, rejected = await batch_score_relevance(
        papers=candidates,
        topic=topic,
        research_questions=research_questions,
        threshold=0.6,
        fallback_threshold=0.5,  # Papers 0.5-0.6 become fallback candidates
        tier=ModelTier.DEEPSEEK_V3,
        max_concurrent=10,
    )

    # Build fallback queue for substitution when acquisition fails
    stage_fallback: list[FallbackCandidate] = [
        FallbackCandidate(
            doi=p.get("doi", ""),
            relevance_score=p.get("relevance_score", 0.5),
            source="near_threshold",
        )
        for p in fallback_candidates
        if p.get("doi")
    ]

    return {
        "current_stage_relevant": [p.get("doi") for p in relevant if p.get("doi")],
        "current_stage_rejected": [p.get("doi") for p in rejected if p.get("doi")],
        "current_stage_fallback": stage_fallback,
    }
```

### Step 5: Implement Dual-Strategy Clustering with Opus Synthesis

Combine BERTopic (statistical) and LLM (semantic) clustering:

```python
# workflows/research/academic_lit_review/clustering/synthesis.py

import json
from typing import Any

from ..state import BERTopicCluster, LLMTopicSchema, ThematicCluster
from workflows.shared.llm_utils import ModelTier, get_structured_output
from .schemas import OpusSynthesisOutput


async def synthesize_clusters_node(state: dict) -> dict[str, Any]:
    """Use Claude Opus to synthesize BERTopic and LLM clustering results.

    Opus reviews both approaches and decides on final clusters.
    Prefers LLM clustering when BERTopic produces poor results.
    """
    bertopic_clusters = state.get("bertopic_clusters", [])
    llm_schema = state.get("llm_topic_schema")
    paper_summaries = state.get("paper_summaries", {})

    # Handle single-source cases
    if not bertopic_clusters and not llm_schema:
        return {"final_clusters": [], "cluster_labels": {}}

    if not bertopic_clusters:
        return _convert_llm_to_final_clusters(llm_schema, paper_summaries)

    if not llm_schema:
        return _convert_bertopic_to_final_clusters(bertopic_clusters, paper_summaries)

    # Evaluate BERTopic quality
    bertopic_good, reason = _evaluate_bertopic_quality(
        bertopic_clusters, len(paper_summaries)
    )

    if not bertopic_good:
        # Skip synthesis, use LLM directly
        return _convert_llm_to_final_clusters(llm_schema, paper_summaries)

    # Both succeeded - use Opus to synthesize
    bertopic_summary = json.dumps([
        {
            "cluster_id": c["cluster_id"],
            "topic_words": c["topic_words"],
            "paper_count": len(c["paper_dois"]),
            "coherence": c["coherence_score"],
        }
        for c in bertopic_clusters
    ], indent=2)

    llm_summary = json.dumps({
        "themes": [
            {
                "name": t["name"],
                "description": t["description"],
                "paper_count": len(t["paper_dois"]),
                "sub_themes": t["sub_themes"],
            }
            for t in llm_schema["themes"]
        ],
        "reasoning": llm_schema["reasoning"],
    }, indent=2)

    result: OpusSynthesisOutput = await get_structured_output(
        output_schema=OpusSynthesisOutput,
        user_prompt=f"Synthesize these clustering results:\n\nBERTopic:\n{bertopic_summary}\n\nLLM:\n{llm_summary}",
        system_prompt="You are synthesizing thematic clusters for a literature review...",
        tier=ModelTier.OPUS,
        max_tokens=16000,
    )

    # Convert to ThematicClusters
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

        for doi in cluster["paper_dois"]:
            cluster_labels[doi] = cluster["cluster_id"]

    return {"final_clusters": final_clusters, "cluster_labels": cluster_labels}
```

### Step 6: Construct the Main Workflow Graph

Connect all phases in a linear flow:

```python
# workflows/research/academic_lit_review/graph/construction.py

from langgraph.graph import END, START, StateGraph

from ..state import AcademicLitReviewState
from .phases import (
    discovery_phase_node,
    diffusion_phase_node,
    processing_phase_node,
    clustering_phase_node,
    synthesis_phase_node,
)


def create_academic_lit_review_graph() -> StateGraph:
    """Create the main academic literature review workflow graph.

    Flow:
        START -> discovery -> diffusion -> processing
              -> clustering -> synthesis -> END

    Note: Supervision loops are added in supervised_lit_review.
    """
    builder = StateGraph(AcademicLitReviewState)

    # Add phase nodes
    builder.add_node("discovery", discovery_phase_node)
    builder.add_node("diffusion", diffusion_phase_node)
    builder.add_node("processing", processing_phase_node)
    builder.add_node("clustering", clustering_phase_node)
    builder.add_node("synthesis", synthesis_phase_node)

    # Linear flow
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", "diffusion")
    builder.add_edge("diffusion", "processing")
    builder.add_edge("processing", "clustering")
    builder.add_edge("clustering", "synthesis")
    builder.add_edge("synthesis", END)

    return builder.compile()


# Export compiled graph
academic_lit_review_graph = create_academic_lit_review_graph()
```

## Consequences

### Benefits

- **Comprehensive discovery**: Citation network expansion finds foundational and related works that keyword search misses
- **Quality-scalable**: Presets allow trading time/cost for thoroughness (50→300 papers)
- **Dual clustering**: BERTopic + LLM + Opus synthesis produces robust thematic organization
- **Two-stage filtering**: Co-citation pre-filtering reduces expensive LLM scoring calls by ~50%
- **Fallback mechanism**: Near-threshold papers (0.5-0.6) substitute when acquisition fails, maintaining corpus size
- **Methodological rigor**: PRISMA documentation satisfies academic publication requirements

### Trade-offs

- **Processing time**: High-quality reviews take 2-4 hours with 300 papers
- **API costs**: Full-text processing + clustering + synthesis = significant LLM usage
- **Diffusion cold start**: Requires good initial seeds (keyword search quality matters)
- **BERTopic sensitivity**: Small corpora (<30 papers) produce poor statistical clusters

### Async Considerations

- **Parallel phases**: Discovery runs keyword, citation, and expert search in parallel via `Send`
- **Batch API**: Processing phase uses Anthropic Batch API for cost savings on large corpora
- **Rate limiting**: OpenAlex API has rate limits; diffusion uses exponential backoff
- **Resource cleanup**: Citation graph cached centrality must be invalidated on changes

## Related Patterns

- [Deep Research Workflow Architecture](./deep-research-workflow-architecture.md) - Simpler web research without citation networks
- [Specialized Researcher Pattern](./specialized-researcher-pattern.md) - Allocation-based researcher dispatch used within discovery phase
- [Multi-Lingual Research Workflow](./multi-lingual-research-workflow.md) - Language handling for non-English sources

## Known Uses in Thala

- `workflows/research/academic_lit_review/graph/construction.py`: Main workflow graph
- `workflows/research/academic_lit_review/citation_graph/builder.py`: NetworkX citation graph
- `workflows/research/academic_lit_review/diffusion_engine/relevance_filters.py`: Two-stage filtering
- `workflows/research/academic_lit_review/clustering/synthesis.py`: Opus cluster synthesis
- `workflows/research/academic_lit_review/quality_presets.py`: Quality tier configuration

## References

- [OpenAlex API Documentation](https://docs.openalex.org/)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [PRISMA Guidelines](http://www.prisma-statement.org/)
- [LangGraph StateGraph](https://python.langchain.com/docs/langgraph/concepts/low_level/#state)
