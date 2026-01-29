---
name: paper-fallback-filtering
title: "Paper Fallback and Intelligent Filtering: Graceful Degradation for Academic Paper Pipelines"
date: 2026-01-28
category: data-pipeline
applicability:
  - "Paper acquisition pipelines with 10-30% failure rates"
  - "Multi-stage discovery with overflow papers exceeding limits"
  - "Content validation against catalog metadata"
components: [async_task, workflow_graph, llm_call]
complexity: complex
verified_in_production: true
related_solutions: []
tags: [fallback, paper-acquisition, validation, co-citation, diffusion, relevance-scoring, graceful-degradation]
---

# Paper Fallback and Intelligent Filtering: Graceful Degradation for Academic Paper Pipelines

## Intent

Enable resilient paper processing pipelines through: (1) fallback substitution for failed acquisitions using near-threshold candidates, (2) content-metadata validation against catalog data, and (3) co-citation context passing to LLMs instead of hard auto-include rules.

## Motivation

Academic paper pipelines face multiple failure modes:

1. **Acquisition failures**: PDFs unavailable, broken OA links, download timeouts
2. **Processing failures**: Invalid PDFs, OCR errors, marker processing failures
3. **Metadata mismatches**: Downloaded PDF doesn't match expected paper
4. **Coverage gaps**: Strict thresholds exclude marginally relevant papers

**Before:**
```
Paper A selected (score=0.75) → Acquisition fails → GAP in output
Paper B (score=0.55) discarded despite being usable fallback
Paper C auto-included (3+ co-citations) despite being off-topic
```

**After:**
```
Paper A fails → Substitute Paper B (score=0.55) from fallback queue
Paper C evaluated by LLM with co-citation count as context → rejected as off-topic
All papers validated against catalog metadata before processing
```

## Applicability

Use this pattern when:
- Paper acquisition has significant failure rates (10-30%)
- Discovery produces more candidates than pipeline capacity
- Papers must match their catalog metadata (prevent wrong-paper errors)
- Graph topology (co-citations) provides relevance signal

Do NOT use this pattern when:
- All papers acquired reliably (no fallback needed)
- Single-paper processing (no queue to manage)
- No catalog metadata available for validation

## Structure

```
Discovery Phase
┌────────────────────────────────────────────────────────────────────────┐
│  batch_score_relevance()                                               │
│    ├─ relevant (≥0.6) → papers_to_process                             │
│    ├─ fallback (0.5-0.6) → fallback_queue (near-threshold)            │
│    └─ rejected (<0.5) → discarded                                     │
│                                                                        │
│  If len(relevant) > max_papers:                                       │
│    overflow = relevant[max_papers:]                                   │
│    fallback_queue += overflow  ← Highest priority fallbacks           │
└────────────────────────────────────────────────────────────────────────┘

Processing Phase
┌────────────────────────────────────────────────────────────────────────┐
│  FallbackManager(fallback_queue, paper_corpus)                         │
│                                                                        │
│  For each paper in papers_to_process:                                 │
│    1. Acquire PDF (OA URL, retrieve-academic, etc.)                   │
│       └─ Failure → get_fallback_for(doi, "acquisition_failed")        │
│                                                                        │
│    2. Process with Marker                                              │
│       └─ Failure → get_fallback_for(doi, "pdf_invalid")               │
│                                                                        │
│    3. Validate content vs catalog metadata                             │
│       ├─ Heuristics: author names, year                               │
│       └─ LLM: semantic check if inconclusive                          │
│       └─ Mismatch → get_fallback_for(doi, "metadata_mismatch")        │
│                                                                        │
│    4. Continue to summarization, synthesis, etc.                      │
└────────────────────────────────────────────────────────────────────────┘

Co-Citation Context (Diffusion Engine)
┌────────────────────────────────────────────────────────────────────────┐
│  enrich_with_cocitation_counts_node()                                 │
│    For each candidate:                                                │
│      count = citation_graph.get_corpus_overlap_count(doi, corpus)     │
│      candidate["corpus_cocitations"] = count                          │
│                                                                        │
│  format_paper_for_batch()                                             │
│    ...                                                                │
│    Corpus Co-citations: 5  ← Passed to LLM as evidence               │
│                                                                        │
│  LLM decides relevance considering co-citations as supporting signal  │
└────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Three-Tier Relevance Scoring

```python
# workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py

async def batch_score_relevance(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float = 0.6,
    fallback_threshold: float = 0.5,
    ...
) -> tuple[list[PaperMetadata], list[PaperMetadata], list[PaperMetadata]]:
    """Score papers and return three-tier categorization.

    Returns:
        Tuple of (relevant_papers, fallback_candidates, rejected_papers)
        - relevant: score >= threshold
        - fallback_candidates: fallback_threshold <= score < threshold
        - rejected: score < fallback_threshold
    """
    # ... scoring logic ...

    for paper, score in zip(papers, scores):
        paper["relevance_score"] = score
        if score >= threshold:
            relevant.append(paper)
        elif score >= fallback_threshold:
            fallback_candidates.append(paper)
        else:
            rejected.append(paper)

    return relevant, fallback_candidates, rejected
```

### Step 2: Fallback Manager

```python
# workflows/research/academic_lit_review/paper_processor/fallback.py

class FallbackManager:
    """Manages fallback paper selection and substitution tracking."""

    def __init__(
        self,
        fallback_queue: list[FallbackCandidate],  # Pre-sorted by relevance
        paper_corpus: dict[str, PaperMetadata],
    ):
        self.queue = list(fallback_queue)
        self.corpus = paper_corpus
        self.used: set[str] = set()
        self.substitutions: list[FallbackSubstitution] = []

    def get_fallback_for(
        self,
        failed_doi: str,
        failure_reason: str,  # "pdf_invalid", "metadata_mismatch", etc.
        failure_stage: str,   # "acquisition", "marker", "validation"
    ) -> Optional[PaperMetadata]:
        """Get next-best fallback paper for a failed DOI."""
        for candidate in self.queue:
            candidate_doi = candidate.get("doi", "")
            if not candidate_doi or candidate_doi in self.used:
                continue

            self.used.add(candidate_doi)

            fallback_metadata = self.corpus.get(candidate_doi)
            if not fallback_metadata:
                continue

            # Record substitution for audit trail
            self.substitutions.append(
                FallbackSubstitution(
                    failed_doi=failed_doi,
                    fallback_doi=candidate_doi,
                    failure_reason=failure_reason,
                    failure_stage=failure_stage,
                )
            )

            logger.info(
                f"Fallback: {failed_doi} -> {candidate_doi} "
                f"(score={candidate.get('relevance_score', 0):.2f})"
            )
            return fallback_metadata

        return None  # Queue exhausted
```

### Step 3: Content-Metadata Validation

```python
# workflows/document_processing/nodes/content_metadata_validator.py

async def validate_content_metadata(state: DocumentProcessingState) -> dict:
    """Validate acquired PDF matches catalog metadata (OpenAlex)."""

    # Use catalog metadata (what we selected by) not extracted metadata
    doc_input = state.get("input", {})
    extra_metadata = doc_input.get("extra_metadata", {})

    metadata = {
        "title": doc_input.get("title"),
        "authors": extra_metadata.get("authors", []),
        "date": extra_metadata.get("date"),
        "venue": extra_metadata.get("publicationTitle"),
    }

    markdown = state["processing_result"]["markdown"]
    content = f"{get_first_n_pages(markdown, 10)}\n...\n{get_last_n_pages(markdown, 10)}"

    # Quick heuristics first
    heuristic_result, confidence, reasoning = _quick_heuristic_check(content, metadata)
    if heuristic_result is True:
        return {"validation_passed": True, "validation_confidence": confidence}

    # LLM semantic check if inconclusive
    result = await get_structured_output(
        output_schema=ContentMetadataMatch,
        user_prompt=f"{content}\n---\nCatalog metadata:\n{metadata_summary}\n---\nTask: Does this content match the catalog metadata?",
        tier=ModelTier.DEEPSEEK_V3,
    )

    if result.matches:
        return {"validation_passed": True, "validation_confidence": result.confidence}
    else:
        return {
            "validation_passed": False,
            "errors": [{"error": f"Mismatch: {result.reasoning}"}],
        }
```

### Step 4: Co-Citation Context Passing

```python
# workflows/research/academic_lit_review/diffusion_engine/relevance_filters.py

async def enrich_with_cocitation_counts_node(state: DiffusionEngineState) -> dict:
    """Add corpus_cocitations field to each candidate for LLM context."""
    candidates = state.get("current_stage_candidates", [])
    citation_graph = state.get("citation_graph")
    corpus_dois = set(state.get("paper_corpus", {}).keys())

    for candidate in candidates:
        doi = candidate.get("doi")
        if doi:
            count = citation_graph.get_corpus_overlap_count(doi, corpus_dois)
            candidate["corpus_cocitations"] = count

    return {"current_stage_candidates": candidates}
```

```python
# workflows/research/academic_lit_review/utils/relevance_scoring/strategies.py

def format_paper_for_batch(paper: PaperMetadata) -> str:
    """Format paper for LLM, including co-citation count if available."""
    lines = [
        f"DOI: {paper.get('doi')}",
        f"Title: {paper.get('title')}",
        f"Authors: {format_authors(paper)}",
        f"Abstract: {paper.get('abstract', '')[:1000]}",
    ]

    # Add co-citations as context signal
    corpus_cocitations = paper.get("corpus_cocitations")
    if corpus_cocitations is not None:
        lines.append(f"Corpus Co-citations: {corpus_cocitations}")

    return "\n".join(lines)
```

## Data Structures

```python
# FallbackCandidate - minimal, pre-scored information
class FallbackCandidate(TypedDict):
    doi: str
    relevance_score: float  # >= 0.5
    source: str  # "overflow" or "near_threshold"

# FallbackSubstitution - audit trail
class FallbackSubstitution(TypedDict):
    failed_doi: str
    fallback_doi: str
    failure_reason: str  # "pdf_invalid", "metadata_mismatch", "acquisition_failed"
    failure_stage: str   # "acquisition", "marker", "validation"
```

## Consequences

### Benefits

- **Graceful degradation**: Pipeline continues despite 20-30% acquisition failures
- **Coverage preservation**: Fallback candidates maintain paper count targets
- **Quality assurance**: Content-metadata validation catches wrong-paper errors
- **Audit trail**: Complete substitution history for post-hoc analysis
- **Nuanced filtering**: LLM considers co-citations as evidence, not hard rule
- **Cost efficiency**: Near-threshold papers reused instead of discarded

### Trade-offs

- **Queue management**: Fallback candidates must propagate through workflow state
- **Validation cost**: LLM calls for content validation (mitigated by heuristics first)
- **Retry overhead**: Failed papers may retry multiple times before substitution
- **Threshold sensitivity**: 0.5-0.6 range may need tuning per domain

### Alternatives

- **No fallback**: Accept gaps in output (simpler but less robust)
- **Auto-include by co-citations**: Hard rules instead of LLM judgment (less nuanced)
- **Retry-only**: Retry failed papers without substitution (slower, may still fail)

## Related Patterns

- [Multi-Strategy Paper Acquisition](../../solutions/api-integration-issues/paper-acquisition-robustness.md) - Acquisition sources
- [Scoring and Regression Detection](../../solutions/llm-issues/scoring-regression-detection.md) - Calibrated scoring
- [Citation Network Academic Review](../langgraph/citation-network-academic-review-workflow.md) - Diffusion architecture

## Known Uses in Thala

- `workflows/research/academic_lit_review/paper_processor/fallback.py` - FallbackManager
- `workflows/document_processing/nodes/content_metadata_validator.py` - Validation
- `workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py` - Three-tier scoring
- `workflows/research/academic_lit_review/diffusion_engine/relevance_filters.py` - Co-citation enrichment
- `workflows/research/academic_lit_review/paper_processor/acquisition/core.py` - Retry integration

## References

- Commit (fallback): a14ab4247daf6f86bd77fe22fef9d494f08507fa
- Commit (co-citation context): 76fa9f8c52c6a5f0588f515c71b2f8f6ff1fef2c
