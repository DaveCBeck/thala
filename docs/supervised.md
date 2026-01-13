# Supervised Lit Review Refactoring

> **Goal:** Make each supervision loop standalone, passing only standardized outputs + original call parameters between them.

## Target Interface

Each loop should accept/return:

```python
# Input (standardized)
{
    "current_review": str,                    # The document
    "topic": str,                             # Original topic
    "research_questions": list[str],          # Original RQs
    "quality": QualityTier,                   # Quality tier
    "source_count": int,                      # Sources so far
    "exclude_dois": set[str],                 # DOIs to skip (for paper fetching)
}

# Output (standardized)
{
    "current_review": str,                    # Modified document
    "changes_summary": str,                   # What changed
    "papers_added": dict[str, PaperSummary],  # New papers (DOI → summary)
    "citations_added": dict[str, str],        # New citations (DOI → zotero_key)
    "issues_flagged": list[dict],             # For human review
}
```

---

## Loop Status

| Loop | Status | Notes |
|------|--------|-------|
| Loop 1 (Theoretical depth) | **Analyzing** | See below |
| Loop 2 (Literature expansion) | Pending | Already has `run_loop2_standalone` |
| Loop 3 (Structure/cohesion) | Pending | Already has `run_loop3_standalone` |
| Loop 4 (Section editing) | Pending | Tight coupling to paper_summaries |
| Loop 4.5 (Cohesion check) | Pending | Already standalone |
| Loop 5 (Fact checking) | Pending | Depends on paper_summaries |

---

## Loop 1 Analysis

**Current inputs consumed:**
- `current_review` - ✓ Keep (core)
- `input` (dict with topic, research_questions) - Flatten to direct params
- `clusters` - Only used for brief summary, **can remove**
- `paper_corpus` - Count for prompt + DOIs for exclusion, simplify to `source_count` + `exclude_dois`
- `paper_summaries` - **Not read**, only merged on output
- `quality_settings` - Only uses `max_stages`, simplify to `max_iterations`
- `zotero_keys` - **Not read**, only merged on output

**Redundant/removable for standalone:**
1. `clusters` parameter - contextual fluff, not essential
2. Full `paper_corpus` - replace with `source_count: int` and `exclude_dois: set[str]`
3. Full `quality_settings` - replace with `max_iterations: int`
4. `paper_summaries` as input - only output new ones

**Impact within Loop 1:**
- `analyze_review_node`: Remove cluster_summary formatting, use source_count directly
- `expand_topic_node`: Accept `exclude_dois` directly instead of extracting from corpus
- `integrate_content_node`: No change needed (already works with expansion results)
- `routing.py`: No change
- `graph.py`/`run_supervision`: Simplify signature, remove unused params

**New standalone signature:**
```python
async def run_loop1_standalone(
    review: str,
    topic: str,
    research_questions: list[str],
    max_iterations: int = 3,
    source_count: int = 0,
    exclude_dois: set[str] | None = None,
) -> Loop1Result
```
