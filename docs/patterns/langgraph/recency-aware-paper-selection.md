---
name: recency-aware-paper-selection
title: "Recency-Aware Paper Selection: Balancing Seminal and Emerging Research"
date: 2026-01-18
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/fa4ef84624fd930bc1d82a73abc91868
article_path: .context/libs/thala-dev/content/2026-01-18-recency-aware-paper-selection-langgraph.md
applicability:
  - "Academic literature reviews requiring coverage of both foundational and recent work"
  - "Citation-based discovery workflows that risk excluding emerging research"
  - "Research fields where recent developments matter significantly"
  - "Quality-tiered workflows needing configurable recency balance"
components: [two_phase_search, recency_quota, collection_overcollect, finalization_partitioning]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [academic-research, citation-filtering, recency-bias, paper-selection, quality-presets, openalex]
---

# Recency-Aware Paper Selection: Balancing Seminal and Emerging Research

## Intent

Ensure literature reviews include both seminal foundational works AND emerging research by using recency-aware filtering that bypasses citation thresholds for recent papers and enforces quota-based selection during finalization.

## Motivation

Citation-based filtering creates a fundamental recency bias:

- **Seminal papers** (10+ years old) have accumulated hundreds of citations
- **Recent papers** (< 3 years) haven't had time to accumulate citations even if important
- A 2024 paper with 5 citations gets filtered out, while a 2014 paper with 50 citations passes
- Result: Literature reviews miss cutting-edge developments

**Example of the problem:**
```
# Query: "machine learning transformers"
# min_citations_filter: 10

✅ Vaswani et al. 2017 "Attention Is All You Need" (50,000+ citations)
✅ Brown et al. 2020 "GPT-3" (8,000+ citations)
❌ Important 2024 paper introducing new technique (8 citations)  ← FILTERED OUT
```

This pattern solves the bias by treating recent and older papers differently.

## Applicability

Use this pattern when:
- Literature review needs to cover both foundational and emerging work
- Citation filtering is used for quality control
- Fast-moving fields where recent breakthroughs matter
- Quality tiers need configurable recency balance

Do NOT use this pattern when:
- Only historical/archival research matters
- No citation filtering is applied
- Corpus is small enough to include all papers
- Recency is not a meaningful quality signal

## Structure

```
                    Quality Settings
                    ┌─────────────────────┐
                    │ recency_years: 3    │
                    │ recency_quota: 0.25 │
                    │ min_citations: 10   │
                    └─────────────────────┘
                           │
    ╔══════════════════════╧══════════════════════╗
    ║          TWO-PHASE DISCOVERY                 ║
    ╠═══════════════════════════════════════════════╣
    ║                                              ║
    ║  ┌──────────────────────────────────────┐   ║
    ║  │        Keyword Search                 │   ║
    ║  │  Phase 1: Recent (>= 2023)            │   ║
    ║  │    → min_citations: 0                 │   ║
    ║  │  Phase 2: Older (< 2023)              │   ║
    ║  │    → min_citations: 10                │   ║
    ║  └──────────────────────────────────────┘   ║
    ║                                              ║
    ║  ┌──────────────────────────────────────┐   ║
    ║  │       Citation Expansion              │   ║
    ║  │  Phase 1: Recent forward citations    │   ║
    ║  │    → min_citations: 0                 │   ║
    ║  │  Phase 2: Older forward citations     │   ║
    ║  │    → min_citations: 10                │   ║
    ║  │  (Backward citations: no filter)      │   ║
    ║  └──────────────────────────────────────┘   ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
                           │
                           ▼
    ╔══════════════════════════════════════════════╗
    ║          OVERCOLLECTION (3x target)          ║
    ╠═══════════════════════════════════════════════╣
    ║                                              ║
    ║  Target: 100 papers                          ║
    ║  Collection target: 300 papers               ║
    ║                                              ║
    ║  Stop when:                                  ║
    ║    - 300 papers collected                    ║
    ║    - Max stages reached                      ║
    ║    - Coverage saturation                     ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
                           │
                           ▼
    ╔══════════════════════════════════════════════╗
    ║          QUOTA-BASED FINALIZATION            ║
    ╠═══════════════════════════════════════════════╣
    ║                                              ║
    ║  Partition: recent vs older                  ║
    ║  Sort: by relevance score                    ║
    ║                                              ║
    ║  Select:                                     ║
    ║    1. Top 25 recent papers (quota)           ║
    ║    2. Top 75 older papers (fill)             ║
    ║    3. Additional recent if slots remain      ║
    ║                                              ║
    ║  Final corpus: 100 papers (25% recent)       ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
```

## Implementation

### Step 1: Add Recency Settings to Quality Presets

```python
# quality_presets.py

from typing import TypedDict


class QualitySettings(TypedDict):
    max_stages: int
    max_papers: int
    target_word_count: int
    min_citations_filter: int
    saturation_threshold: float
    use_batch_api: bool
    supervision_loops: str
    recency_years: int        # NEW: Years to consider "recent"
    recency_quota: float      # NEW: Target fraction of recent papers


QUALITY_PRESETS: dict[str, QualitySettings] = {
    "test": QualitySettings(
        max_stages=1,
        max_papers=5,
        target_word_count=500,
        min_citations_filter=0,
        saturation_threshold=0.5,
        use_batch_api=True,
        supervision_loops="all",
        recency_years=3,
        recency_quota=0.0,  # Skip recency quota for test tier
    ),
    "quick": QualitySettings(
        max_stages=2,
        max_papers=30,
        target_word_count=3000,
        min_citations_filter=5,
        saturation_threshold=0.15,
        use_batch_api=True,
        supervision_loops="all",
        recency_years=3,
        recency_quota=0.25,  # 25% recent papers
    ),
    "standard": QualitySettings(
        max_stages=3,
        max_papers=50,
        target_word_count=6000,
        min_citations_filter=10,
        saturation_threshold=0.12,
        use_batch_api=True,
        supervision_loops="all",
        recency_years=3,
        recency_quota=0.25,
    ),
    # comprehensive and high_quality follow same pattern...
}
```

### Step 2: Two-Phase Keyword Search

```python
# keyword_search/searcher.py

from datetime import datetime, timezone


async def search_openalex_node(state: KeywordSearchState) -> dict[str, Any]:
    """Execute searches with recency-aware citation thresholds."""
    quality_settings = state["quality_settings"]
    queries = state["queries"]

    min_citations = quality_settings.get("min_citations_filter", 10)
    recency_years = quality_settings.get("recency_years", 3)

    current_year = datetime.now(timezone.utc).year
    recent_cutoff = current_year - recency_years  # e.g., 2023 if current is 2026

    async def search_single_query(query: str) -> list[OpenAlexWork]:
        """Search with two-phase approach."""
        works = []

        # Phase 1: Recent papers with NO citation filter
        recent_result = await openalex_search.ainvoke({
            "query": query,
            "limit": MAX_RESULTS_PER_QUERY,
            "min_citations": 0,  # Key: No threshold for recent
            "from_year": recent_cutoff,
            "to_year": None,
        })
        works.extend(recent_result.get("results", []))

        # Phase 2: Older papers with normal citation threshold
        older_result = await openalex_search.ainvoke({
            "query": query,
            "limit": MAX_RESULTS_PER_QUERY,
            "min_citations": min_citations,  # Normal threshold
            "from_year": None,
            "to_year": recent_cutoff - 1,
        })
        works.extend(older_result.get("results", []))

        return works

    # Execute all queries
    results = await asyncio.gather(*[search_single_query(q) for q in queries])

    return {"raw_results": [r for batch in results for r in batch]}
```

### Step 3: Two-Phase Citation Fetching

```python
# diffusion_engine/citation_fetcher.py

from datetime import datetime, timezone


async def fetch_citations_raw(
    seed_dois: list[str],
    min_citations: int = 10,
    recency_years: int = 3,
) -> tuple[list[dict], list[CitationEdge]]:
    """Fetch forward citations with recency-aware thresholds."""
    current_year = datetime.now(timezone.utc).year
    recent_cutoff = current_year - recency_years

    async def fetch_single_paper(seed_doi: str):
        forward_papers = []
        edges = []
        seen_dois: set[str] = set()

        # Phase 1: Recent forward citations (no threshold)
        recent_forward = await get_forward_citations(
            work_id=seed_doi,
            limit=MAX_CITATIONS_PER_PAPER,
            min_citations=0,  # No threshold for recent
            from_year=recent_cutoff,
        )

        for work in recent_forward.results:
            doi = work.get("doi", "")
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                forward_papers.append(work)
                edges.append(CitationEdge(
                    citing_doi=doi,
                    cited_doi=seed_doi,
                    edge_type="forward",
                ))

        # Phase 2: Older forward citations (with threshold)
        older_forward = await get_forward_citations(
            work_id=seed_doi,
            limit=MAX_CITATIONS_PER_PAPER,
            min_citations=min_citations,  # Normal threshold
        )

        for work in older_forward.results:
            doi = work.get("doi", "")
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                forward_papers.append(work)
                edges.append(CitationEdge(
                    citing_doi=doi,
                    cited_doi=seed_doi,
                    edge_type="forward",
                ))

        # Backward citations: No recency filter needed (historical references)
        backward_result = await get_backward_citations(work_id=seed_doi)
        # ... (standard backward citation handling)

        return forward_papers, backward_papers, edges

    # ... gather all results
```

### Step 4: Overcollection Target

```python
# diffusion_engine/termination.py


async def check_saturation_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Check if diffusion should continue."""
    max_papers = state["quality_settings"]["max_papers"]
    paper_corpus = state.get("paper_corpus", {})

    # KEY: Collect 3x max_papers to ensure enough recent papers for quota
    collection_target = max_papers * 3

    saturation_reason = None

    if state["diffusion"]["current_stage"] >= state["diffusion"]["max_stages"]:
        saturation_reason = f"Reached maximum stages"

    elif len(paper_corpus) >= collection_target:
        saturation_reason = f"Reached collection target ({collection_target} for {max_papers} final)"

    elif state["diffusion"]["consecutive_low_coverage"] >= 2:
        saturation_reason = "Coverage saturation"

    if saturation_reason:
        return {"saturation_reason": saturation_reason}
    else:
        return {}  # Continue diffusion
```

### Step 5: Quota-Based Finalization

```python
# diffusion_engine/termination.py

from datetime import datetime, timezone


async def finalize_diffusion(state: DiffusionEngineState) -> dict[str, Any]:
    """Finalize with recency quota enforcement."""
    paper_corpus = state.get("paper_corpus", {})
    quality_settings = state["quality_settings"]
    max_papers = quality_settings["max_papers"]
    recency_years = quality_settings.get("recency_years", 3)
    recency_quota = quality_settings.get("recency_quota", 0.25)

    # If corpus is small enough, no filtering needed
    if len(paper_corpus) <= max_papers:
        return {"final_corpus_dois": list(paper_corpus.keys())}

    # Partition by recency
    current_year = datetime.now(timezone.utc).year
    cutoff_year = current_year - recency_years

    recent = [(doi, p) for doi, p in paper_corpus.items() if p.get("year", 0) >= cutoff_year]
    older = [(doi, p) for doi, p in paper_corpus.items() if p.get("year", 0) < cutoff_year]

    # Sort each by relevance score
    recent.sort(key=lambda x: x[1].get("relevance_score", 0.5), reverse=True)
    older.sort(key=lambda x: x[1].get("relevance_score", 0.5), reverse=True)

    # Calculate target for recent papers
    target_recent = int(max_papers * recency_quota)

    # Select: recent first (up to quota), then older to fill
    recent_selected = recent[: min(target_recent, len(recent))]
    remaining_slots = max_papers - len(recent_selected)
    older_selected = older[:remaining_slots]

    # If extra slots and more recent papers, add them
    total_selected = len(recent_selected) + len(older_selected)
    if total_selected < max_papers and len(recent) > len(recent_selected):
        extra_needed = max_papers - total_selected
        recent_selected.extend(recent[len(recent_selected):len(recent_selected) + extra_needed])

    final_dois = [doi for doi, _ in recent_selected] + [doi for doi, _ in older_selected]

    # Log composition
    actual_recent = len([d for d in final_dois if paper_corpus[d].get("year", 0) >= cutoff_year])
    recent_pct = actual_recent / len(final_dois) if final_dois else 0
    logger.info(
        f"Diffusion complete: {len(final_dois)} papers "
        f"({actual_recent} recent = {recent_pct:.0%})"
    )

    return {"final_corpus_dois": final_dois}
```

## Complete Example

```python
from workflows.research.academic_lit_review import academic_lit_review

# Standard quality with 25% recency quota
result = await academic_lit_review(
    topic="Transformer architectures in NLP",
    quality="standard",  # 50 papers, 25% recent
)

# Result includes balanced corpus
print(f"Total papers: {len(result['final_corpus_dois'])}")  # 50

# Check recency distribution
recent = [d for d in result['final_corpus_dois']
          if result['paper_corpus'][d].get('year', 0) >= 2023]
print(f"Recent papers (2023+): {len(recent)}")  # ~12-13 (25%)
print(f"Older papers: {50 - len(recent)}")      # ~37-38 (75%)
```

## Consequences

### Benefits

- **Balanced coverage**: Both seminal and emerging work included
- **No recency bias**: Recent papers aren't penalized for low citations
- **Configurable**: Quality tiers can adjust recency quota
- **Transparent**: Final corpus composition is logged

### Trade-offs

- **Overcollection cost**: 3x papers collected means more API calls
- **Quality dilution risk**: Recent papers may have less rigorous peer review
- **Quota rigidity**: Fixed 25% may not suit all topics
- **Complexity**: Two-phase search/fetch adds implementation complexity

### Alternatives

- **Manual curation**: Human selects recent papers separately
- **Year-weighted scoring**: Boost relevance scores for recent papers
- **Separate pools**: Maintain independent recent/seminal corpora

## Related Patterns

- [Citation Network Academic Review Workflow](./citation-network-academic-review-workflow.md) - Full workflow using this pattern
- [Unified Quality Tier System](./unified-quality-tier-system.md) - Quality presets structure
- [Researcher Allocation and Query Optimization](./researcher-allocation-query-optimization.md) - Dynamic search allocation

## Known Uses in Thala

- `workflows/research/academic_lit_review/quality_presets.py` - Quality settings with recency_years/recency_quota
- `workflows/research/academic_lit_review/keyword_search/searcher.py` - Two-phase keyword search
- `workflows/research/academic_lit_review/diffusion_engine/citation_fetcher.py` - Two-phase citation fetching
- `workflows/research/academic_lit_review/diffusion_engine/termination.py` - Quota-based finalization

## References

- [OpenAlex API Documentation](https://docs.openalex.org/)
- [Citation Analysis in Systematic Reviews](https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.MR000024.pub3/full)
