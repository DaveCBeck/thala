# Academic Literature Review Workflow Plan

## Executive Summary

This document outlines the architecture for a new **Academic Literature Review Workflow** - a parallel, multi-stage system designed to produce PhD-equivalent literature reviews (10-15k words) by discovering, reading, and synthesizing 200+ academic sources through recursive diffusion.

The workflow transforms the existing `academic_researcher` from a simple search-and-summarize component into a sophisticated **citation network exploration system** that:

1. Takes a research question/topic
2. Finds seminal and recent papers through multiple discovery strategies
3. Builds a citation network through forward/backward snowballing
4. Recursively expands coverage through diffusion stages
5. Clusters and organizes sources thematically
6. Synthesizes findings into a coherent, properly-cited literature review

---

## 1. Vision & Goals

### 1.1 Target Output

On "high quality" setting, the system produces:
- **Length**: 10-15k words
- **Sources**: 200+ academic papers, books, and grey literature
- **Depth**: PhD-equivalent systematic literature review
- **Structure**: Thematic organization with clear argumentation
- **Citations**: Properly formatted academic citations with full metadata

### 1.2 Key Differentiators

| Current Academic Researcher | New Literature Review Workflow |
|---------------------------|-------------------------------|
| Single search iteration | Multi-stage recursive diffusion |
| 3-5 papers per question | 200+ sources through snowballing |
| Individual paper summaries | Thematic synthesis across sources |
| No citation network awareness | Forward/backward citation exploration |
| No saturation detection | Automatic coverage completeness |

### 1.3 Design Principles

1. **Diffusion-based expansion**: Start from seed papers, expand via citations
2. **Multi-strategy discovery**: Combine keyword search, citation network, and expert identification
3. **Recursive depth**: Allow multiple "rounds" of discovery until saturation
4. **Thematic clustering**: Organize sources by theme, not just chronology
5. **Quality over quantity**: Prioritize seminal and high-impact papers
6. **Grounded synthesis**: Every claim backed by specific citations

---

## 2. Architecture Overview

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ACADEMIC LITERATURE REVIEW WORKFLOW                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT: Research Question/Topic                                             │
│      │                                                                       │
│      ▼                                                                       │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 1: DISCOVERY (Parallel)                                        │  │
│   │                                                                      │  │
│   │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │  │
│   │   │  Keyword   │  │  Citation  │  │   Expert   │  │    Book    │   │  │
│   │   │  Search    │  │  Network   │  │ Identifier │  │   Search   │   │  │
│   │   │ (OpenAlex) │  │ (Snowball) │  │ (H-Index)  │  │(book_search)│  │  │
│   │   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   │  │
│   │         │               │               │               │          │  │
│   │         └───────────────┴───────┬───────┴───────────────┘          │  │
│   │                                 ▼                                   │  │
│   │                    ┌───────────────────────┐                        │  │
│   │                    │   Seed Paper Pool     │                        │  │
│   │                    │   (Deduplicated)      │                        │  │
│   │                    └───────────┬───────────┘                        │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 2: DIFFUSION (Recursive)                                       │  │
│   │                                                                      │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │                  DIFFUSION STAGE N                          │   │  │
│   │   │                                                             │   │  │
│   │   │   Seed Papers ──► Forward Citation ──► New Papers           │   │  │
│   │   │        │              (Who cited?)          │               │   │  │
│   │   │        │                                    │               │   │  │
│   │   │        └────► Backward Citation ──► New Papers              │   │  │
│   │   │                    (References)                             │   │  │
│   │   │                                                             │   │  │
│   │   │   Relevance Filter ──► Add to Corpus ──► Check Saturation  │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                         │                                           │  │
│   │                         ▼                                           │  │
│   │              ┌────────────────────┐                                 │  │
│   │              │ Saturation Check   │──── Not Saturated ─┐            │  │
│   │              │ (Diminishing new   │                    │            │  │
│   │              │  relevant papers)  │                    │            │  │
│   │              └─────────┬──────────┘                    │            │  │
│   │                        │ Saturated                     │            │  │
│   │                        ▼                               │            │  │
│   │              ┌────────────────────┐       ┌────────────┴──────┐    │  │
│   │              │   Paper Corpus     │◄──────│ Next Diffusion    │    │  │
│   │              │   (200+ papers)    │       │ Stage (max 5)     │    │  │
│   │              └─────────┬──────────┘       └───────────────────┘    │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 3: PROCESSING (Parallel Batches)                               │  │
│   │                                                                      │  │
│   │   Paper Corpus ──► Batch Scrape/Process ──► Paper Summaries         │  │
│   │                    (Marker for PDFs)        (Structured metadata)    │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 4: ORGANIZATION (Clustering)                                   │  │
│   │                                                                      │  │
│   │   Paper Summaries ──► Topic Modeling ──► Thematic Clusters          │  │
│   │                       (BERTopic)          (5-15 themes)              │  │
│   │                                                                      │  │
│   │   Per Cluster: Identify key papers, conflicts, gaps                 │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 5: SYNTHESIS (Recursive Writing)                               │  │
│   │                                                                      │  │
│   │   Thematic Clusters ──► Section Drafts ──► Integration ──► Review   │  │
│   │                                                                      │  │
│   │   Structure:                                                         │  │
│   │   1. Executive Summary                                               │  │
│   │   2. Introduction & Scope                                            │  │
│   │   3. Methodology (Search Strategy)                                   │  │
│   │   4. Thematic Sections (per cluster)                                 │  │
│   │   5. Discussion & Gaps                                               │  │
│   │   6. Conclusions                                                     │  │
│   │   7. References                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│   OUTPUT: 10-15k Word Literature Review + Citation Database                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Summary

| Phase | Component | Purpose | Parallelism |
|-------|-----------|---------|-------------|
| 1 | Discovery | Find initial seed papers | 4 parallel strategies |
| 2 | Diffusion | Expand via citations | Per-paper parallel expansion |
| 3 | Processing | **document_processing workflow** + retrieve-academic | Batch API (50% savings) |
| 4 | Organization | **Dual clustering** (BERTopic + Sonnet 4.5 1M) → **Opus synthesis** | 2 parallel strategies |
| 5 | Synthesis | Write the review + **Zotero citations** | Section-parallel, then sequential |

---

## 3. Detailed Component Design

### 3.1 Phase 1: Discovery Subgraphs

#### 3.1.1 Keyword Search Subgraph

```python
# File: workflows/research/subgraphs/academic_lit_review/keyword_search.py

class KeywordSearchState(TypedDict):
    """State for keyword-based academic search."""
    topic: str
    search_queries: list[str]           # Generated academic queries
    raw_results: list[OpenAlexWork]      # From OpenAlex
    filtered_results: list[SeedPaper]    # After relevance filtering

async def generate_academic_queries(state) -> dict:
    """Generate multiple query variations for comprehensive coverage.

    Query strategies:
    1. Exact topic + methodology terms
    2. Broader field + specific concepts
    3. Related terminology / synonyms
    4. Historical + recent framing
    """

async def search_openalex(state) -> dict:
    """Execute searches with filters for quality.

    Filters:
    - cited_by_count > 10 (avoid noise)
    - publication_year ranges (recent + seminal)
    - is_oa = true preferred (for full text)
    """

async def filter_by_relevance(state) -> dict:
    """LLM-based relevance scoring."""
```

**Key Features:**
- Multiple query reformulations per topic
- Date-range stratification (recent 2 years + seminal 10+ years)
- Quality filters (citation count, open access, peer-reviewed)
- Relevance scoring via LLM

#### 3.1.2 Citation Network Subgraph (Initial)

```python
# File: workflows/research/subgraphs/academic_lit_review/citation_network.py

class CitationNetworkState(TypedDict):
    """State for citation-based discovery."""
    seed_dois: list[str]                 # Starting papers (from keyword search)
    forward_citations: list[OpenAlexWork]  # Papers citing seeds
    backward_citations: list[OpenAlexWork] # References in seeds
    network_edges: list[CitationEdge]      # For graph construction

async def fetch_forward_citations(state) -> dict:
    """Find papers that cite the seed papers.

    Uses OpenAlex cited_by endpoint.
    Prioritizes by citation count and recency.
    """

async def fetch_backward_citations(state) -> dict:
    """Find papers referenced by seed papers.

    Uses OpenAlex referenced_works endpoint.
    Identifies frequently co-cited "must-read" papers.
    """
```

**OpenAlex API Usage:**
```python
# Forward citations (who cited this paper?)
GET /works/{work_id}/cited_by

# Backward citations (what did this paper cite?)
# Available in work response: referenced_works field
GET /works/{work_id}?select=id,referenced_works
```

#### 3.1.3 Expert Identifier Subgraph

```python
# File: workflows/research/subgraphs/academic_lit_review/expert_identifier.py

class ExpertIdentifierState(TypedDict):
    """State for finding domain experts."""
    topic: str
    candidate_authors: list[OpenAlexAuthor]
    expert_papers: list[OpenAlexWork]      # Top papers from experts

async def identify_field_experts(state) -> dict:
    """Find prolific/influential authors in the field.

    Metrics:
    - Works count in topic area
    - Citation count (h-index proxy)
    - Recent activity (still publishing)
    - Institutional affiliation

    Uses OpenAlex Authors API with topic filtering.
    """

async def fetch_expert_works(state) -> dict:
    """Get top-cited works from identified experts."""
```

**OpenAlex Author API:**
```python
# Find authors by topic
GET /authors?filter=topics.id:{topic_id}&sort=cited_by_count:desc

# Get author's works
GET /works?filter=authorships.author.id:{author_id}&sort=cited_by_count:desc
```

#### 3.1.4 Book Search Subgraph

```python
# File: workflows/research/subgraphs/academic_lit_review/book_discovery.py

class BookDiscoveryState(TypedDict):
    """State for finding relevant books."""
    topic: str
    book_queries: list[str]
    books: list[Book]

async def generate_book_queries(state) -> dict:
    """Generate queries optimized for book discovery.

    Focus on:
    - "Handbook of X"
    - "Introduction to X"
    - "Companion to X"
    - Author names (from expert identification)
    - Classic/foundational texts
    """
```

### 3.2 Phase 2: Diffusion Engine

The core innovation: **recursive citation network expansion with saturation detection**.

#### 3.2.1 Diffusion State

```python
# File: workflows/research/subgraphs/academic_lit_review/state.py

class DiffusionStage(TypedDict):
    """State for a single diffusion stage."""
    stage_number: int
    seed_papers: list[str]               # DOIs to expand from
    forward_papers: list[OpenAlexWork]    # New papers from forward citations
    backward_papers: list[OpenAlexWork]   # New papers from backward citations
    new_relevant: list[str]              # DOIs passing relevance filter
    coverage_delta: float                # How much new ground covered

class LitReviewDiffusionState(TypedDict):
    """Main diffusion algorithm state."""
    topic: str
    research_questions: list[str]

    # Paper corpus (accumulates across stages)
    paper_corpus: dict[str, PaperMetadata]  # DOI -> metadata
    paper_summaries: dict[str, str]          # DOI -> summary

    # Diffusion tracking
    current_stage: int
    max_stages: int                          # Default: 5
    stages: list[DiffusionStage]

    # Saturation detection
    saturation_threshold: float              # Default: 0.1 (10% new relevant)
    is_saturated: bool

    # Citation network
    citation_graph: CitationGraph            # For visualization & analysis
```

#### 3.2.2 Diffusion Algorithm

```python
# File: workflows/research/subgraphs/academic_lit_review/diffusion_engine.py

async def run_diffusion_stage(state: LitReviewDiffusionState) -> dict:
    """Execute one diffusion stage.

    Algorithm:
    1. Select seed papers for this stage
       - Stage 0: Use discovery phase results
       - Stage N>0: Use papers added in stage N-1

    2. For each seed paper (parallel):
       a. Fetch forward citations (who cited this?)
       b. Fetch backward citations (what did this cite?)
       c. Deduplicate against existing corpus

    3. Determine relevance (two-stage):
       a. Co-citation check: papers co-cited with 3+ corpus papers → auto-include
       b. LLM scoring: remaining papers scored using abstract (or full-text summary)

    4. Add relevant papers to corpus

    5. Calculate coverage delta
       - coverage_delta = new_relevant / total_candidates
       - If < saturation_threshold, mark saturated

    6. Build citation edges for graph
    """

async def select_expansion_seeds(state) -> list[str]:
    """Select which papers to expand from.

    Prioritization:
    1. Highly cited papers (seminal works)
    2. Recent papers with growing citations (rising stars)
    3. Papers with many unexplored citations
    4. Papers central to citation network (high betweenness)
    """

async def determine_relevance(
    candidate_papers: list[OpenAlexWork],
    corpus: dict[str, PaperMetadata],
    topic: str,
) -> list[tuple[str, float]]:
    """Two-stage relevance determination.

    Stage 1: Citation-network signals (structural)
    - Papers frequently co-cited with existing corpus papers → auto-include
    - Uses co-citation count threshold (e.g., co-cited with 3+ corpus papers)
    - Fast, no LLM cost, leverages network structure

    Stage 2: LLM classification (semantic) - for remaining papers
    - Uses abstract if available
    - If no abstract: generate summary from full text via retrieve-academic
    - Returns relevance score 0-1
    - Threshold: > 0.6 to include

    Returns (doi, relevance_score) pairs.
    """

def check_cocitation_relevance(
    paper_doi: str,
    corpus_dois: set[str],
    citation_graph: CitationGraph,
    threshold: int = 3,
) -> bool:
    """Check if paper is frequently co-cited with corpus papers.

    A paper is considered structurally relevant if it shares citations
    with multiple existing corpus papers (bibliographic coupling) or
    is cited alongside them (co-citation).

    This captures papers that are clearly part of the same literature
    without needing semantic analysis.
    """

async def llm_relevance_scoring(
    papers: list[OpenAlexWork],
    topic: str,
) -> list[tuple[str, float]]:
    """LLM-based relevance for papers not caught by co-citation.

    For each paper:
    1. Use abstract if available
    2. If no abstract: fetch full text via retrieve-academic, generate summary
    3. Score relevance to topic (0-1)

    Batched for efficiency.
    """

def check_saturation(state) -> bool:
    """Determine if diffusion should stop.

    Saturation conditions (any triggers stop):
    1. coverage_delta < saturation_threshold for 2 consecutive stages
    2. max_stages reached
    3. paper_corpus size exceeds max_papers (default: 300)
    """
```

#### 3.2.3 Citation Graph Construction

```python
# File: workflows/research/subgraphs/academic_lit_review/citation_graph.py

class CitationGraph:
    """In-memory citation network for analysis."""

    def __init__(self):
        self.nodes: dict[str, PaperNode] = {}  # DOI -> node
        self.edges: list[CitationEdge] = []     # (citing, cited) pairs

    def add_paper(self, doi: str, metadata: PaperMetadata):
        """Add paper as node."""

    def add_citation(self, citing_doi: str, cited_doi: str):
        """Add directed edge."""

    def get_seminal_papers(self, top_n: int = 10) -> list[str]:
        """Find most-cited papers in the network.

        Uses in-degree (citation count within network).
        """

    def get_bridging_papers(self, top_n: int = 10) -> list[str]:
        """Find papers connecting different clusters.

        Uses betweenness centrality.
        """

    def get_recent_impactful(self, years: int = 3, top_n: int = 10) -> list[str]:
        """Find recent papers with high citation velocity.

        Citations per year since publication.
        """

    def identify_clusters(self) -> list[list[str]]:
        """Community detection for thematic grouping.

        Uses Louvain algorithm on co-citation network.
        """
```

### 3.3 Phase 3: Paper Processing (via document_processing workflow)

**Key Design Decision**: Leverage the existing `document_processing` workflow instead of building custom processing. This provides:
- Full PDF → markdown conversion via Marker
- Automatic Elasticsearch indexing (3 compression levels)
- Automatic Zotero library management
- 50% cost savings via Anthropic Batch API for large corpora
- Existing summarization + metadata extraction

#### 3.3.1 Integration with document_processing

```python
# File: workflows/research/subgraphs/academic_lit_review/paper_processor.py

from workflows.document_processing import process_documents_with_batch_api
from core.stores.retrieve_academic import RetrieveAcademicClient

class PaperProcessingState(TypedDict):
    """State for paper processing."""
    paper_queue: list[PaperMetadata]     # Papers to process
    elasticsearch_ids: dict[str, str]    # DOI -> ES record ID
    zotero_keys: dict[str, str]          # DOI -> Zotero key
    processed: dict[str, PaperSummary]   # DOI -> summary (from ES)
    failed: list[str]                    # DOIs that failed

async def acquire_full_text(paper: PaperMetadata) -> tuple[str, str]:
    """Acquire full text for a paper using retrieve-academic service.

    Does NOT prefer OA - retrieves any paper via DOI or title search.
    The retrieve-academic service handles paywalled content.

    Returns:
        Tuple of (local_file_path, file_format)
    """
    async with RetrieveAcademicClient() as client:
        # Check service health
        if not await client.health_check():
            raise RuntimeError("Retrieve-academic service unavailable")

        # Submit retrieval request
        local_path, result = await client.retrieve_and_download(
            doi=paper.doi,
            local_path=f"/tmp/papers/{paper.doi.replace('/', '_')}.pdf",
            title=paper.title,
            authors=[a.name for a in paper.authors],
            timeout=180.0,  # 3 min for larger papers
        )

        return local_path, result.file_format

async def process_paper_corpus(
    papers: list[PaperMetadata],
    use_batch_api: bool = True,
) -> dict[str, PaperSummary]:
    """Process entire paper corpus through document_processing workflow.

    Two-stage process:
    1. Acquire full text for all papers via retrieve-academic
    2. Batch process through document_processing (50% LLM cost savings)

    After processing, papers are:
    - Stored in Elasticsearch (full text + summaries)
    - Indexed in Zotero library
    - Ready for PaperSummary extraction
    """
    # Stage 1: Acquire full text (parallel with rate limiting)
    acquired_papers = []
    semaphore = asyncio.Semaphore(5)  # Limit concurrent retrievals

    async def acquire_with_limit(paper):
        async with semaphore:
            try:
                path, fmt = await acquire_full_text(paper)
                return {"paper": paper, "path": path, "format": fmt}
            except Exception as e:
                logger.warning(f"Failed to acquire {paper.doi}: {e}")
                return None

    results = await asyncio.gather(*[acquire_with_limit(p) for p in papers])
    acquired_papers = [r for r in results if r is not None]

    # Stage 2: Process via document_processing batch API
    if use_batch_api and len(acquired_papers) > 20:
        # Use batch API for 50% cost savings
        batch_results = await process_documents_with_batch_api(
            documents=[
                {
                    "id": p["paper"].doi,
                    "source": p["path"],
                    "title": p["paper"].title,
                }
                for p in acquired_papers
            ],
            include_metadata=True,
            include_chapter_summaries=False,  # Papers typically don't need this
        )
    else:
        # Use standard processing for smaller batches
        from workflows.document_processing import process_documents_batch
        batch_results = await process_documents_batch(
            documents=[...],
            concurrency=5,
        )

    # Stage 3: Extract PaperSummary from processed documents in ES
    return await extract_paper_summaries_from_es(batch_results)

async def extract_paper_summaries_from_es(
    processing_results: dict,
) -> dict[str, PaperSummary]:
    """Generate PaperSummary objects from processed documents in Elasticsearch.

    The document_processing workflow already created:
    - compression_level=0: Full markdown content
    - compression_level=1: 100-word short summary
    - compression_level=2: 10:1 summary (for long papers)

    This function extracts structured PaperSummary for clustering/synthesis.
    """
    store_manager = get_store_manager()
    paper_summaries = {}

    for doi, result in processing_results.items():
        # Get full content from ES (compression_level=0)
        full_record = await store_manager.es_stores.store.get(
            result["store_records"][0].id
        )

        # Extract structured summary via LLM
        summary = await extract_structured_summary(
            content=full_record.content,
            metadata=result.get("metadata", {}),
            short_summary=result.get("short_summary", ""),
        )

        paper_summaries[doi] = summary

    return paper_summaries
```

#### 3.3.2 PaperSummary Extraction

```python
class PaperSummary(TypedDict):
    """Structured summary of a paper - extracted from ES after processing."""
    doi: str
    title: str
    authors: list[str]
    year: int
    venue: str

    # From document_processing
    short_summary: str                   # 100-word summary (compression_level=1)
    es_record_id: str                    # Reference to full content
    zotero_key: str                      # Reference to Zotero item

    # Extracted for clustering/synthesis
    key_findings: list[str]              # 3-5 main findings
    methodology: str                      # Research method summary
    limitations: list[str]               # Stated limitations
    future_work: list[str]               # Suggested future directions
    themes: list[str]                    # Topic tags for clustering
    claims: list[ClaimWithEvidence]      # Extractable claims

async def extract_structured_summary(
    content: str,
    metadata: dict,
    short_summary: str,
) -> PaperSummary:
    """Extract structured fields from processed paper content.

    Uses the full markdown content (from ES compression_level=0)
    to extract key findings, methodology, limitations, etc.

    This runs AFTER document_processing, using the stored content.
    """
    # LLM extraction prompt for academic papers
    extraction_prompt = PAPER_SUMMARY_EXTRACTION_PROMPT.format(
        content=content[:50000],  # First ~50k chars
        metadata=json.dumps(metadata),
        short_summary=short_summary,
    )

    result = await extract_json_cached(
        text=content,
        prompt=extraction_prompt,
        schema_hint=PAPER_SUMMARY_SCHEMA,
        tier=ModelTier.SONNET,
    )

    return PaperSummary(**result)

### 3.4 Phase 4: Organization (Dual-Strategy Clustering)

**Key Design Decision**: Use PARALLEL clustering strategies, then have Opus synthesize the final theme structure. This ensures we get both:
- Statistical rigor (BERTopic embedding-based clustering)
- Semantic intelligence (LLM-based thematic analysis with 1M context)

#### 3.4.1 Dual Clustering Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: CLUSTERING                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Paper Summaries (200+)                                             │
│          │                                                           │
│          ├──────────────────────┬──────────────────────┐            │
│          ▼                      ▼                      │            │
│   ┌──────────────┐      ┌──────────────┐               │            │
│   │  BERTopic    │      │ Claude Sonnet│               │            │
│   │  Clustering  │      │ 4.5 (1M ctx) │               │            │
│   │              │      │              │               │            │
│   │ • Embedding  │      │ • All 200+   │               │            │
│   │ • UMAP       │      │   summaries  │               │            │
│   │ • HDBSCAN    │      │   in context │               │            │
│   │ • Topic repr │      │ • Semantic   │               │            │
│   └──────┬───────┘      │   grouping   │               │            │
│          │              └──────┬───────┘               │            │
│          │                     │                       │            │
│          ▼                     ▼                       │            │
│   ┌──────────────┐      ┌──────────────┐               │            │
│   │ Statistical  │      │  LLM Topic   │               │            │
│   │  Clusters    │      │   Schema     │               │            │
│   │ (5-15 groups)│      │ (5-15 themes)│               │            │
│   └──────┬───────┘      └──────┬───────┘               │            │
│          │                     │                       │            │
│          └─────────┬───────────┘                       │            │
│                    ▼                                   │            │
│          ┌─────────────────┐                           │            │
│          │   Claude Opus   │                           │            │
│          │   Synthesis     │                           │            │
│          │                 │                           │            │
│          │ • Compare both  │                           │            │
│          │ • Merge/refine  │                           │            │
│          │ • Final themes  │                           │            │
│          └────────┬────────┘                           │            │
│                   ▼                                    │            │
│          ┌─────────────────┐                           │            │
│          │ Final Thematic  │                           │            │
│          │   Clusters      │                           │            │
│          └─────────────────┘                           │            │
│                   │                                    │            │
│                   ▼                                    │            │
│          Per-Cluster Analysis (parallel)               │            │
│                                                        │            │
└────────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Clustering State

```python
# File: workflows/research/subgraphs/academic_lit_review/clustering.py

class ClusteringState(TypedDict):
    """State for dual-strategy thematic clustering."""
    paper_summaries: dict[str, PaperSummary]

    # Parallel clustering results
    bertopic_clusters: list[BERTopicCluster]     # Statistical clustering
    llm_topic_schema: LLMTopicSchema              # Semantic clustering

    # Synthesized result
    final_clusters: list[ThematicCluster]
    cluster_labels: dict[str, int]               # DOI -> cluster_id

class BERTopicCluster(TypedDict):
    """Output from BERTopic statistical clustering."""
    cluster_id: int
    topic_words: list[str]                       # Representative words
    paper_dois: list[str]
    coherence_score: float

class LLMTopicSchema(TypedDict):
    """Output from LLM semantic clustering."""
    themes: list[LLMTheme]
    reasoning: str                               # LLM's clustering rationale

class LLMTheme(TypedDict):
    """A theme identified by the LLM."""
    name: str
    description: str
    paper_dois: list[str]
    sub_themes: list[str]
    relationships: list[str]                     # How this theme relates to others

class ThematicCluster(TypedDict):
    """Final synthesized thematic grouping."""
    cluster_id: int
    label: str                                   # Final theme name
    description: str                             # What this cluster covers
    paper_dois: list[str]
    key_papers: list[str]                        # Most central papers
    sub_themes: list[str]                        # Finer-grained topics
    conflicts: list[str]                         # Contradictory findings
    gaps: list[str]                              # Under-researched areas
    source: str                                  # "bertopic", "llm", or "merged"
```

#### 3.4.3 Parallel Clustering Implementation

```python
async def cluster_papers_parallel(state: ClusteringState) -> dict:
    """Run BERTopic and LLM clustering in parallel, then synthesize.

    This is the main clustering orchestrator.
    """
    paper_summaries = state["paper_summaries"]

    # Run both clustering strategies in parallel
    bertopic_task = asyncio.create_task(run_bertopic_clustering(paper_summaries))
    llm_task = asyncio.create_task(run_llm_clustering(paper_summaries))

    bertopic_clusters, llm_schema = await asyncio.gather(bertopic_task, llm_task)

    # Synthesize with Opus
    final_clusters = await synthesize_clusters_with_opus(
        bertopic_clusters=bertopic_clusters,
        llm_schema=llm_schema,
        paper_summaries=paper_summaries,
    )

    return {
        "bertopic_clusters": bertopic_clusters,
        "llm_topic_schema": llm_schema,
        "final_clusters": final_clusters,
        "cluster_labels": {
            doi: cluster.cluster_id
            for cluster in final_clusters
            for doi in cluster["paper_dois"]
        },
    }

async def run_bertopic_clustering(
    paper_summaries: dict[str, PaperSummary],
) -> list[BERTopicCluster]:
    """Statistical clustering using BERTopic.

    Process:
    1. Create document representations (title + abstract + key_findings)
    2. Embed documents using sentence-transformers
    3. Reduce dimensionality with UMAP
    4. Cluster with HDBSCAN
    5. Extract topic representations
    """
    from bertopic import BERTopic

    # Prepare documents
    docs = [
        f"{s['title']}\n{s.get('abstract', '')}\n{' '.join(s.get('key_findings', []))}"
        for s in paper_summaries.values()
    ]
    dois = list(paper_summaries.keys())

    # Run BERTopic
    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        min_topic_size=5,
        nr_topics="auto",  # Let it determine optimal number
    )
    topics, probs = topic_model.fit_transform(docs)

    # Build cluster output
    clusters = []
    for topic_id in set(topics):
        if topic_id == -1:  # Skip outliers
            continue

        cluster_dois = [dois[i] for i, t in enumerate(topics) if t == topic_id]
        topic_info = topic_model.get_topic(topic_id)

        clusters.append(BERTopicCluster(
            cluster_id=topic_id,
            topic_words=[word for word, _ in topic_info[:10]],
            paper_dois=cluster_dois,
            coherence_score=probs[topics == topic_id].mean(),
        ))

    return clusters

async def run_llm_clustering(
    paper_summaries: dict[str, PaperSummary],
) -> LLMTopicSchema:
    """Semantic clustering using Claude Sonnet 4.5 with 1M context.

    Feeds ALL paper summaries to a single LLM call, leveraging the
    1M token context window to see the entire corpus at once.

    This allows the LLM to:
    - Understand nuanced relationships between papers
    - Identify themes that statistical methods might miss
    - Create semantically meaningful groupings
    """
    # Prepare all summaries for the prompt
    summaries_text = "\n\n---\n\n".join([
        f"DOI: {doi}\n"
        f"Title: {s['title']}\n"
        f"Authors: {', '.join(s['authors'])}\n"
        f"Year: {s['year']}\n"
        f"Abstract: {s.get('abstract', 'N/A')}\n"
        f"Key Findings: {', '.join(s.get('key_findings', []))}\n"
        f"Methodology: {s.get('methodology', 'N/A')}"
        for doi, s in paper_summaries.items()
    ])

    llm = get_llm(
        ModelTier.SONNET,
        model_id="claude-sonnet-4-5-20241022",  # Sonnet 4.5 with 1M context
        max_tokens=16000,
    )
    structured_llm = llm.with_structured_output(LLMTopicSchema)

    prompt = LLM_CLUSTERING_PROMPT.format(
        paper_count=len(paper_summaries),
        summaries=summaries_text,
    )

    result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
    return result

LLM_CLUSTERING_PROMPT = """You are analyzing {paper_count} academic papers to identify thematic clusters.

Your task is to organize these papers into 5-15 coherent themes based on:
- Research topics and questions
- Methodological approaches
- Theoretical frameworks
- Application domains
- Temporal developments

For each theme, identify:
- A clear, descriptive name
- A 2-3 sentence description
- Which paper DOIs belong to this theme
- Sub-themes if the cluster is broad
- How this theme relates to other themes

Papers may belong to multiple themes if they bridge topics.

Here are all the paper summaries:

{summaries}

Analyze these papers and output a structured topic schema."""

async def synthesize_clusters_with_opus(
    bertopic_clusters: list[BERTopicCluster],
    llm_schema: LLMTopicSchema,
    paper_summaries: dict[str, PaperSummary],
) -> list[ThematicCluster]:
    """Use Claude Opus to compare and synthesize clustering results.

    Opus reviews both approaches and decides:
    - Which clusters to keep from each
    - Which to merge
    - Which to split
    - Final theme names and descriptions
    """
    llm = get_llm(ModelTier.OPUS)
    structured_llm = llm.with_structured_output(FinalClusteringDecision)

    prompt = OPUS_SYNTHESIS_PROMPT.format(
        bertopic_summary=json.dumps([c for c in bertopic_clusters], indent=2),
        llm_schema_summary=json.dumps(llm_schema, indent=2),
        paper_count=len(paper_summaries),
    )

    decision = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
    return decision.final_clusters

OPUS_SYNTHESIS_PROMPT = """You are synthesizing two different clustering analyses of {paper_count} academic papers.

## Statistical Clustering (BERTopic)
Based on embedding similarity and density:
{bertopic_summary}

## Semantic Clustering (LLM Analysis)
Based on conceptual understanding:
{llm_schema_summary}

Your task:
1. Compare the two clustering approaches
2. Identify where they agree and disagree
3. Decide on a final set of 5-15 themes that best organizes the literature
4. For each final theme:
   - Choose the best name (may be from either source or your own)
   - Write a clear description
   - Assign papers (DOIs) to this theme
   - Identify key/central papers
   - Note any sub-themes
   - Flag potential conflicts between papers
   - Identify gaps in the research

Prefer semantic coherence over statistical purity. The goal is themes that
support writing a coherent literature review.

Output your final clustering decision."""
```

#### 3.4.4 Per-Cluster Analysis

```python
async def generate_cluster_analysis(cluster: ThematicCluster, summaries: dict) -> dict:
    """Deep analysis of a single cluster.

    Produces:
    - Narrative summary of the theme
    - Timeline of developments
    - Key debates and positions
    - Methodological approaches used
    - Outstanding questions
    """
    cluster_summaries = {doi: summaries[doi] for doi in cluster["paper_dois"]}

    llm = get_llm(ModelTier.SONNET)

    analysis = await llm.ainvoke([
        {"role": "system", "content": CLUSTER_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": format_cluster_for_analysis(cluster, cluster_summaries)},
    ])

    return {
        "narrative_summary": analysis.narrative_summary,
        "timeline": analysis.timeline,
        "key_debates": analysis.key_debates,
        "methodologies": analysis.methodologies,
        "outstanding_questions": analysis.outstanding_questions,
    }
```

### 3.5 Phase 5: Synthesis (Writing)

#### 3.5.1 Literature Review Structure

```python
# File: workflows/research/subgraphs/academic_lit_review/synthesis.py

LITERATURE_REVIEW_STRUCTURE = """
# [Topic] Literature Review

## Abstract
[200-300 words summarizing the review]

## 1. Introduction
### 1.1 Background and Rationale
### 1.2 Research Questions
### 1.3 Scope and Boundaries

## 2. Methodology
### 2.1 Search Strategy
[Document search terms, databases, date ranges]
### 2.2 Selection Criteria
[Inclusion/exclusion criteria]
### 2.3 Data Extraction
[What was extracted from each paper]
### 2.4 Synthesis Approach
[How findings were integrated]

## 3. [Theme 1 from clustering]
### 3.1 Overview
### 3.2 Key Findings
### 3.3 Methodological Approaches
### 3.4 Debates and Tensions
### 3.5 Gaps and Limitations

## 4. [Theme 2 from clustering]
[Same structure as Theme 1]

... [Repeat for each cluster, 3-8 themes typically]

## N. Discussion
### N.1 Synthesis of Findings
### N.2 Implications
### N.3 Limitations of This Review
### N.4 Future Research Directions

## N+1. Conclusions
[Key takeaways, 500-800 words]

## References
[Full citations in consistent format]

## Appendix: Search Documentation
[Full PRISMA-style documentation]
"""
```

#### 3.5.2 Section Writing Pipeline

```python
class SynthesisState(TypedDict):
    """State for literature review writing."""
    clusters: list[ThematicCluster]
    paper_summaries: dict[str, PaperSummary]

    # Writing progress
    section_drafts: dict[str, str]       # Section name -> draft
    integrated_draft: str                 # Full merged document
    final_review: str                     # After revision

    # Quality tracking
    citation_coverage: float              # % of corpus cited
    claim_support: dict[str, list[str]]  # Claim -> supporting DOIs

async def write_thematic_section(cluster: ThematicCluster, summaries: dict) -> str:
    """Write a single thematic section.

    Process:
    1. Retrieve all paper summaries for cluster
    2. Identify narrative arc (chronological, conceptual, methodological)
    3. Generate section outline
    4. Write subsections with inline citations
    5. Add synthesis paragraphs connecting ideas
    6. Verify all claims have citations
    """

async def integrate_sections(sections: dict[str, str]) -> str:
    """Merge sections into coherent document.

    Process:
    1. Write introduction framing all themes
    2. Add transition paragraphs between sections
    3. Write methodology section documenting search
    4. Write discussion synthesizing across themes
    5. Write conclusions
    6. Compile references (deduplicated, formatted)
    """

async def revise_and_verify(draft: str, summaries: dict) -> str:
    """Quality assurance pass.

    Checks:
    1. Citation accuracy (every claim has source)
    2. No hallucinated citations
    3. Balanced coverage across corpus
    4. Coherent argumentation
    5. Proper academic tone
    """
```

---

## 4. Quality Settings & Resource Allocation

### 4.1 Quality Tiers

| Setting | Max Stages | Max Papers | Target Length | Typical Duration |
|---------|-----------|------------|---------------|------------------|
| Quick | 2 | 50 | 3-5k words | 15-30 min |
| Standard | 3 | 100 | 5-8k words | 30-60 min |
| Comprehensive | 4 | 200 | 8-12k words | 1-2 hours |
| **High Quality** | 5 | 300 | 10-15k words | 2-4 hours |

### 4.2 Resource Allocation by Phase

| Phase | Parallelism | API Calls (High Quality) | LLM Calls |
|-------|------------|--------------------------|-----------|
| Discovery | 4 parallel | ~20 OpenAlex | 4 query generation |
| Diffusion | 10 papers/batch | ~150 OpenAlex | 15 relevance batches |
| Processing | Batch API (5 concurrent retrieval) | ~200 retrieve-academic | Via document_processing batch API |
| Clustering | 2 parallel (BERTopic + Sonnet 4.5) | 0 | 1 Sonnet 1M + 1 Opus synthesis |
| Synthesis | 5 sections parallel | 0 | 20+ writing calls + citation processing |

### 4.3 Cost Estimation (High Quality)

```
OpenAlex API: Free (with polite rate limits)
Retrieve-academic service: Local (free, handles paywalled papers)
Marker PDF processing: Local (free)

LLM Costs (using Anthropic Batch API - 50% discount):
- Document processing (200 papers): ~400k tokens × $0.0015 = $0.60
- PaperSummary extraction: ~200k tokens × $0.0015 = $0.30
- Dual clustering (Sonnet 4.5 1M): ~300k tokens × $0.0015 = $0.45
- Opus cluster synthesis: ~50k tokens × $0.0075 = $0.38
- Section writing (Sonnet): ~400k tokens × $0.0015 = $0.60
- Final synthesis (Opus): ~100k tokens × $0.0075 = $0.75

Total estimated LLM cost: ~$3.08 (with batch API)
Total estimated LLM cost: ~$6.16 (without batch API)

Total estimated cost per high-quality review: $3-4 (batch) / $6-8 (realtime)
```

**Key Savings**:
- 50% LLM cost reduction via Anthropic Batch API
- Zero scraping costs (retrieve-academic handles full-text)
- Local Marker processing (no cloud PDF conversion)

---

## 5. State Schema

### 5.1 Main Workflow State

```python
# File: workflows/research/subgraphs/academic_lit_review/state.py

class AcademicLitReviewState(TypedDict):
    """Complete state for literature review workflow."""

    # Input
    input: LitReviewInput

    # Discovery phase results
    keyword_papers: list[str]            # DOIs from keyword search
    citation_papers: list[str]           # DOIs from initial citation search
    expert_papers: list[str]             # DOIs from expert identification
    books: list[BookMetadata]            # Books found

    # Diffusion tracking
    diffusion: LitReviewDiffusionState

    # Paper corpus
    paper_corpus: Annotated[dict[str, PaperMetadata], merge_dicts]
    paper_summaries: Annotated[dict[str, PaperSummary], merge_dicts]
    citation_graph: CitationGraph

    # Clustering
    clusters: list[ThematicCluster]

    # Synthesis
    section_drafts: dict[str, str]
    final_review: str

    # Output
    references: list[FormattedCitation]
    prisma_documentation: str            # Search methodology

    # Metadata
    started_at: datetime
    completed_at: datetime
    current_phase: str
    errors: Annotated[list[dict], add]

class LitReviewInput(TypedDict):
    """Input parameters for literature review."""
    topic: str
    research_questions: list[str]        # Specific questions to address
    quality: Literal["quick", "standard", "comprehensive", "high_quality"]

    # Optional constraints
    date_range: tuple[int, int] | None   # (start_year, end_year)
    include_books: bool                  # Default: True
    focus_areas: list[str] | None        # Specific sub-topics to prioritize
    exclude_terms: list[str] | None      # Terms to filter out
```

---

## 6. Integration with Existing System

### 6.1 Relationship to Current Workflow

The new `academic_lit_review` workflow is a **separate, specialized workflow** that can be:

1. **Invoked directly** for dedicated literature review tasks
2. **Called from the main research workflow** when deep academic coverage is needed
3. **Used to feed the existing academic_researcher** with better seed papers

```python
# Option 1: Direct invocation
from workflows.research.academic_lit_review import academic_lit_review

result = await academic_lit_review(
    topic="transformer architectures in NLP",
    research_questions=[
        "How have attention mechanisms evolved since 2017?",
        "What are the computational efficiency improvements?",
    ],
    quality="high_quality"
)

# Option 2: From main research workflow (when supervisor detects need)
# In supervisor.py, add detection for "literature review" requests
if requires_comprehensive_academic_review(research_brief):
    return {
        "current_status": "literature_review",
        "lit_review_params": {...}
    }
```

### 6.2 Shared Components

Reuse from existing codebase:

| Component | Current Location | Reuse Strategy |
|-----------|-----------------|----------------|
| OpenAlex search | `langchain_tools/openalex.py` | Extend with citation endpoints |
| Book search | `langchain_tools/book_search.py` | Use directly |
| **Document processing** | `workflows/document_processing/` | **Use for all paper processing (50% cost savings via batch API)** |
| **Retrieve-academic** | `core/stores/retrieve_academic.py` | **Use for full-text acquisition (handles paywalled papers)** |
| **Zotero integration** | `core/stores/zotero.py` | **Use for reference management (auto-creates library items)** |
| **Citation processing** | `workflows/research/nodes/process_citations.py` | **Converts to Pandoc [@KEY] format** |
| Marker PDF | `workflows/shared/marker_client.py` | Used by document_processing |
| Scrape cache | `workflows/research/subgraphs/researcher_base.py` | Share singleton |
| LLM utilities | `workflows/shared/llm_utils.py` | Use directly |
| Prompt caching | `invoke_with_cache()` | Apply to all synthesis |

### 6.3 Key Integration Points

#### Document Processing Integration

```python
# Papers flow through document_processing for:
# 1. PDF → Markdown (via Marker)
# 2. Elasticsearch indexing (3 compression levels)
# 3. Zotero item creation
# 4. Summary + metadata extraction

from workflows.document_processing import process_documents_with_batch_api

# 50% cost savings on 200+ papers
results = await process_documents_with_batch_api(
    documents=[{"source": pdf_path, "title": title} for ...],
    include_metadata=True,
)
```

#### Zotero Reference Management

```python
# All papers automatically get Zotero items via document_processing
# Final report uses existing process_citations node to:
# 1. Convert [1], [2] → [@ZOTERO_KEY]
# 2. Generate proper bibliography

from workflows.research.nodes.process_citations import process_citations

# After synthesis, process citations for proper formatting
final_state = await process_citations(state_with_report)
# final_state["final_report"] now has Pandoc-style citations
# final_state["citation_keys"] has all Zotero keys
```

### 6.4 New Dependencies

```python
# requirements.txt additions
bertopic>=0.16.0          # Topic modeling (dual clustering)
networkx>=3.0             # Citation graph analysis
python-louvain>=0.16      # Community detection
sentence-transformers     # Embeddings for BERTopic clustering
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create `workflows/research/subgraphs/academic_lit_review/` directory structure
- [ ] Implement `state.py` with all TypedDict definitions
- [ ] Extend `langchain_tools/openalex.py` with citation endpoints:
  - `get_forward_citations(doi, limit)`
  - `get_backward_citations(doi, limit)`
  - `get_author_works(author_id, limit)`
- [ ] Create `citation_graph.py` with basic graph operations

### Phase 2: Discovery Subgraphs (Week 2)
- [ ] Implement `keyword_search.py` subgraph
- [ ] Implement `citation_network.py` initial discovery
- [ ] Implement `expert_identifier.py` subgraph
- [ ] Integrate `book_discovery.py` (light wrapper around existing)
- [ ] Create discovery coordinator node

### Phase 3: Diffusion Engine (Week 3)
- [ ] Implement `diffusion_engine.py` core algorithm
- [ ] Add saturation detection logic
- [ ] Add seed selection prioritization
- [ ] Implement batch relevance scoring
- [ ] Add citation graph construction during diffusion

### Phase 4: Paper Processing Integration (Week 4)
- [ ] Create `paper_processor.py` that wraps document_processing workflow
- [ ] Integrate with `retrieve-academic` service for full-text acquisition
- [ ] Create `PaperSummary` extraction prompts (runs after document_processing)
- [ ] Add batch processing with progress callbacks
- [ ] Implement fallback for failed retrievals (abstract-only mode)

### Phase 5: Dual Clustering & Synthesis (Week 5)
- [ ] Implement `clustering.py` with dual strategy:
  - [ ] BERTopic statistical clustering
  - [ ] Sonnet 4.5 (1M context) semantic clustering
  - [ ] Opus synthesis node
- [ ] Create cluster analysis prompts
- [ ] Implement `synthesis.py` section writing
- [ ] Integrate with `process_citations` node for Zotero-backed references
- [ ] Add quality verification pass

### Phase 6: Integration & Testing (Week 6)
- [ ] Create main `graph.py` connecting all phases
- [ ] Add quality tier configurations
- [ ] Create test suite with sample topics
- [ ] Add PRISMA documentation generation
- [ ] Performance optimization (batch API, parallelism)
- [ ] End-to-end test with 50+ papers

---

## 8. Example Usage

```python
from workflows.research.academic_lit_review import academic_lit_review

# High-quality literature review
result = await academic_lit_review(
    topic="Large Language Models in Scientific Discovery",
    research_questions=[
        "How are LLMs being used for hypothesis generation?",
        "What are the methodological challenges of using LLMs in research?",
        "What domains show the most promising LLM applications?",
    ],
    quality="high_quality",
    date_range=(2020, 2025),
    include_books=True,
    focus_areas=["biology", "chemistry", "materials science"],
)

# Access results
print(f"Papers analyzed: {len(result['paper_corpus'])}")
print(f"Themes identified: {len(result['clusters'])}")
print(f"Review length: {len(result['final_review'].split())} words")

# All papers are now:
# - Indexed in Elasticsearch (full text + summaries)
# - Added to Zotero library with full metadata
print(f"Zotero items created: {len(result['zotero_keys'])}")
print(f"Elasticsearch records: {len(result['elasticsearch_ids'])}")

# Save outputs
with open("literature_review.md", "w") as f:
    # Report has Pandoc-style citations: [@ZOTERO_KEY]
    f.write(result['final_review'])

# Generate bibliography from Zotero
# Option 1: Export from Zotero directly (via CSL styles)
# Option 2: Use citation keys for Pandoc processing
print(f"Citation keys for Pandoc: {result['citation_keys']}")

# The review can now be processed with Pandoc:
# pandoc literature_review.md --citeproc --bibliography=zotero.bib -o review.pdf
```

### Zotero Integration Details

```python
# After processing, all papers are in Zotero with:
# - Full metadata (title, authors, date, venue)
# - Abstract (100-word summary from document_processing)
# - Tags: ["thala-research", "auto-citation", topic_tag]
# - DOI/URL links

# Access papers in Zotero programmatically:
from core.stores.zotero import ZoteroStore

async with ZoteroStore() as zotero:
    # Search by tag
    lit_review_items = await zotero.search_by_tag("thala-research")

    # Get full item details
    for item in lit_review_items:
        full_item = await zotero.get(item.key)
        print(f"{full_item.fields.get('title')}: {full_item.key}")
```

---

## 9. Success Metrics

### 9.1 Quantitative Metrics

| Metric | Target (High Quality) |
|--------|----------------------|
| Papers discovered | 200-300 |
| Papers cited in review | 150+ |
| Themes identified | 5-15 |
| Final word count | 10,000-15,000 |
| Citation accuracy | 100% (verified) |
| Coverage of seminal works | 90%+ |

### 9.2 Quality Indicators

- [ ] No hallucinated citations
- [ ] Every claim backed by specific source
- [ ] Balanced representation across themes
- [ ] Identifies key debates and tensions
- [ ] Explicitly notes gaps and limitations
- [ ] Proper academic tone and structure
- [ ] Coherent narrative across sections

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OpenAlex rate limits | Slow discovery | Use polite pool (email), implement backoff |
| PDF access (paywalls) | Incomplete coverage | Use retrieve-academic service (handles any paper via DOI/title) |
| LLM hallucinations | Incorrect citations | Verification pass, grounded prompts, Zotero citation keys |
| Token limits | Truncated synthesis | Chunking strategy, hierarchical writing |
| Diffusion explosion | Too many papers | Hard caps, strict relevance threshold |
| Clustering quality | Poor themes | Dual strategy (BERTopic + LLM), Opus synthesis |
| Batch API delays | Slow processing | Progress callbacks, hybrid approach for urgent papers |

---

## Appendix A: OpenAlex API Reference

### Key Endpoints for Literature Review

```python
# Search works
GET /works?search={query}&filter={filters}&sort={sort}

# Get work by ID (includes references)
GET /works/{id}?select=id,doi,title,referenced_works

# Get citing works
GET /works?filter=cites:{work_id}

# Get author's works
GET /works?filter=authorships.author.id:{author_id}

# Get topics
GET /topics?search={query}

# Get works by topic
GET /works?filter=topics.id:{topic_id}
```

### Useful Filters

```python
# Quality filters
cited_by_count:>10              # Minimum citations
is_oa:true                      # Open access only
type:journal-article            # Peer-reviewed

# Date filters
publication_year:>2020          # Recent
publication_year:2015-2020      # Range

# Content filters
abstract.search:{query}         # Abstract contains
title.search:{query}            # Title contains
```

---

## Appendix B: Prompts Reference

### Key Prompts to Develop

1. **Query Generation**: Academic search query optimization
2. **Relevance Scoring**: Paper-topic relevance assessment
3. **Paper Summarization**: Structured extraction from full text
4. **Cluster Labeling**: Theme naming from paper groups
5. **Cluster Analysis**: Identifying debates, gaps, key papers
6. **Section Writing**: Academic prose with inline citations
7. **Section Integration**: Coherent document assembly
8. **Quality Verification**: Citation and claim checking

See `workflows/research/prompts/academic_lit_review/` for full prompt templates.
