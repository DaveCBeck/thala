---
module: academic_lit_review
date: 2026-01-04
problem_type: state_management_bug
component: diffusion, processing, focused_expansion
symptoms:
  - "Quick mode (max 50 papers) processing 300+ papers"
  - "Processing phase takes longer than expected for quality tier"
  - "max_papers limit not being respected"
  - "All discovered papers processed instead of filtered subset"
root_cause: state_field_ignored
resolution_type: state_propagation_fix
severity: high
tags: [quality-settings, state-management, workflow, max-papers, filtering]
---

# Quality Setting Propagation: max_papers

## Problem

The literature review workflow ignored the `max_papers` quality setting during processing. "Quick" mode with `max_papers=50` was processing 300+ papers:

```
Starting processing phase for 347 papers  # Should be ≤50
```

This caused quick reviews to take hours instead of minutes.

## Root Cause

**Downstream phases used full corpus instead of filtered list.**

The diffusion phase filters papers to `max_papers`, storing the filtered DOIs in `final_corpus_dois`. However, downstream phases iterated over `paper_corpus.keys()` (all discovered papers) instead of using the filtered list:

```python
# PROBLEMATIC: Uses full corpus, ignores max_papers filtering
papers_to_process = list(paper_corpus.values())  # 347 papers

# Should use filtered list
papers_to_process_dois = state.get("papers_to_process")  # 50 papers
```

Three locations had this bug:
1. **diffusion_phase_node**: Set `papers_to_process` to all corpus keys
2. **processing_phase_node**: Iterated `paper_corpus.values()` directly
3. **focused_expansion**: Used `expanded_corpus.items()` instead of filtered DOIs

## Solution

**Propagate filtered DOI list through state and use it consistently.**

### Fix 1: Diffusion Phase Sets Filtered List

```python
# workflows/research/subgraphs/academic_lit_review/graph/phases/diffusion.py

async def diffusion_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    # ... diffusion execution ...

    final_corpus = diffusion_result.get("paper_corpus", paper_corpus)
    # Get the filtered DOI list, not all keys
    final_corpus_dois = diffusion_result.get(
        "final_corpus_dois",
        list(final_corpus.keys())
    )

    logger.info(
        f"Diffusion complete: {len(final_corpus_dois)} papers selected from "
        f"{len(final_corpus)} discovered. Reason: {saturation_reason}"
    )

    return {
        "paper_corpus": final_corpus,
        "diffusion": diffusion_state,
        # Pass filtered list, not all keys
        "papers_to_process": final_corpus_dois,
        "current_status": f"Diffusion complete: {len(final_corpus_dois)} papers",
    }
```

### Fix 2: Processing Phase Uses State Field

```python
# workflows/research/subgraphs/academic_lit_review/graph/phases/processing.py

async def processing_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    paper_corpus = state.get("paper_corpus", {})

    # Use filtered papers_to_process list (set by diffusion), not full corpus
    papers_to_process_dois = state.get(
        "papers_to_process",
        list(paper_corpus.keys())  # Fallback for backward compat
    )
    papers_to_process = [
        paper_corpus[doi]
        for doi in papers_to_process_dois
        if doi in paper_corpus
    ]

    logger.info(
        f"Starting processing phase for {len(papers_to_process)} papers "
        f"(filtered from {len(paper_corpus)} discovered)"
    )

    processing_result = await run_paper_processing(
        papers=papers_to_process,  # Filtered list
        quality_settings=quality_settings,
    )
```

### Fix 3: Focused Expansion Uses Filtered DOIs

```python
# workflows/research/subgraphs/academic_lit_review/supervision/focused_expansion.py

async def run_focused_expansion(...) -> dict[str, Any]:
    # ... diffusion for expansion ...

    expanded_corpus = diffusion_result.get("paper_corpus", paper_corpus)
    # Get filtered list from diffusion (respects max_papers)
    final_corpus_dois = diffusion_result.get(
        "final_corpus_dois",
        list(expanded_corpus.keys())
    )

    # Filter to new papers AND in the filtered list
    final_corpus = {
        doi: expanded_corpus[doi]
        for doi in final_corpus_dois
        if doi in expanded_corpus and doi not in exclude_dois
    }

    logger.info(
        f"Diffusion expanded to {len(expanded_corpus)} papers, "
        f"filtered to {len(final_corpus_dois)}, {len(final_corpus)} are new"
    )
```

## State Flow

```
Discovery Phase
  └─► paper_corpus: {doi1, doi2, ..., doi347}  # All discovered

Diffusion Phase
  └─► paper_corpus: {doi1, doi2, ..., doi347}  # Kept for reference
  └─► final_corpus_dois: [doi1, doi5, ..., doi89]  # Filtered to max_papers
  └─► papers_to_process: [doi1, doi5, ..., doi89]  # Passed downstream

Processing Phase
  └─► Uses papers_to_process (50 papers)
  └─► NOT paper_corpus.keys() (347 papers)
```

## Files Modified

- `workflows/research/subgraphs/academic_lit_review/graph/phases/diffusion.py`
- `workflows/research/subgraphs/academic_lit_review/graph/phases/processing.py`
- `workflows/research/subgraphs/academic_lit_review/supervision/focused_expansion.py`

## Prevention

When passing filtered subsets through workflow state:

1. **Use dedicated state field**: Create `papers_to_process` instead of deriving from corpus keys
2. **Document the contract**: State field docstring should specify it's filtered
3. **Log both counts**: "Processing X papers (filtered from Y discovered)"
4. **Test with assertions**: Verify `len(papers_to_process) <= quality_settings.max_papers`

## Testing

```python
async def test_processing_respects_max_papers():
    """Verify processing phase uses filtered list, not full corpus."""
    state = {
        "paper_corpus": {f"doi_{i}": {} for i in range(300)},  # 300 papers
        "papers_to_process": [f"doi_{i}" for i in range(50)],  # Filtered to 50
        "quality_settings": {"max_papers": 50},
    }

    result = await processing_phase_node(state)

    # Should only process 50 papers
    assert len(result["paper_summaries"]) <= 50
```

## Related Patterns

- [Quality Tier Standardization](../../patterns/langgraph/quality-tier-standardization.md) - Quality settings design
- [Iterative Theoretical Depth Supervision](../../patterns/langgraph/iterative-theoretical-depth-supervision.md) - Uses focused_expansion
