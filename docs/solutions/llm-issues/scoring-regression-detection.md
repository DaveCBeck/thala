---
module: workflows/research/academic_lit_review, workflows/enhance/editing
date: 2026-01-28
problem_type: llm_issue
component: llm_call
symptoms:
  - "Inconsistent relevance scores across large paper batches (calibration drift)"
  - "Document coherence degraded after structural edits despite passing verification"
  - "Independent paper scoring produced inconsistent score distributions"
  - "Structural edits introduced logical gaps not caught by standard verification"
root_cause: logic_error
resolution_type: code_fix
severity: medium
verified_fix: true
tags: [scoring, calibration, regression-detection, coherence, rollback, cross-comparison, sonnet]
---

# Scoring and Regression Detection Solutions

## Problem

Two related quality scoring issues were degrading workflow reliability:

**1. Relevance Score Calibration Drift:**
When scoring papers individually, the model's perception of "0.7 relevance" varied without context. A paper might score 0.7 in isolation but deserve 0.5 when compared to more relevant papers in the same domain.

**2. Coherence Regression from Structural Edits:**
The editing workflow could degrade document coherence despite "improving" structure. Edits that fix organizational issues (reordering sections, consolidating content) might inadvertently break narrative flow or create logical gaps.

**Symptoms:**
```
Before (calibration):
  Paper A (isolated) → 0.7
  Paper B (isolated) → 0.7
  Paper C (isolated) → 0.7
  Reality: A=0.8, B=0.5, C=0.6 (no relative comparison)

Before (regression):
  Original document coherence: 0.85
  After structural edits: 0.78
  Workflow proceeded → degraded output went to polish phase
```

## Solution

### Part 1: Chunked Cross-Comparison Scoring

Group papers into chunks of 10 for batch scoring, enabling cross-comparison calibration.

**Implementation** (`scorer.py`):

```python
async def _batch_score_deepseek_cached(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float,
    fallback_threshold: float,
    language_config: LanguageConfig | None,
    tier: ModelTier,
    max_concurrent: int,
    chunk_size: int = 10,  # Papers per chunk for cross-comparison
) -> tuple[list[PaperMetadata], list[PaperMetadata], list[PaperMetadata]]:
    """Score papers with cross-comparison within chunks."""

    # Chunk papers for comparative scoring
    chunks = chunk_papers(papers, chunk_size)

    # Build cache prefix (shared across all chunks)
    cache_prefix = f"Research Topic: {topic}\nResearch Questions: {research_questions_str}\n\nPapers to Evaluate:"

    # Build user prompts for each chunk (10 papers per prompt)
    user_prompts: list[tuple[str, str]] = []
    for i, chunk in enumerate(chunks):
        papers_text = "\n".join(format_paper_for_batch(p) for p in chunk)
        user_prompt = BATCH_RELEVANCE_SCORING_USER_TEMPLATE.format(
            topic=topic,
            research_questions=research_questions_str,
            papers=papers_text,  # All 10 papers together
        )
        user_prompts.append((f"chunk-{i}", user_prompt))

    # Process with cache warmup coordination
    responses = await batch_invoke_with_cache(
        llm,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        cache_prefix=cache_prefix,
        max_concurrent=max_concurrent,
    )

    # Parse JSON array response (one score per paper)
    for chunk_id, papers_in_chunk in chunk_index.items():
        response = responses.get(chunk_id)
        parsed = json.loads(response.content)

        # Match DOIs to scores
        doi_scores = {item["doi"]: item["relevance_score"] for item in parsed}
        for paper in papers_in_chunk:
            paper["relevance_score"] = doi_scores.get(paper["doi"], 0.5)
```

**Benefits:**
- Papers scored relative to 9 others in same topic
- LLM makes comparative judgments: "This paper is less relevant than these 4 papers"
- ~90% cost reduction on chunks 2-N (cache warmup pattern)
- Three-tier output: relevant, fallback_candidates, rejected

### Part 2: Coherence Regression Detection with Rollback

Detect when structural edits degrade coherence and automatically recover.

**Step 1: Capture Baseline** (`analyze_structure.py`):

```python
if iteration == 0:
    # Capture baseline coherence on first iteration
    baseline_coherence = (
        analysis.narrative_coherence_score +
        analysis.section_organization_score
    ) / 2
    return {"baseline_coherence_score": baseline_coherence, ...}
```

**Step 2: Detect Regression** (`verify_structure.py`):

```python
REGRESSION_THRESHOLD = 0.05  # 5% drop triggers investigation

baseline_coherence = state.get("baseline_coherence_score")
if baseline_coherence is not None:
    coherence_drop = baseline_coherence - verification.coherence_score

    if coherence_drop > REGRESSION_THRESHOLD:
        # Call Sonnet to compare both document versions
        comparison = await _check_coherence_regression(
            original_model=original_model,
            edited_model=updated_model,
            topic=topic,
            edits_summary=edits_summary,
        )

        if comparison and comparison.preferred_version == "original" and comparison.confidence >= 0.6:
            # Confirmed regression
            if not retry_used:
                # First regression: rollback and retry
                return {
                    "updated_document_model": state["document_model"],  # Rollback
                    "coherence_regression_retry_used": True,
                    "structure_iteration": iteration,  # Don't increment!
                    "needs_more_structure_work": True,  # Retry
                }
            else:
                # Second regression: give up, proceed with original
                return {
                    "updated_document_model": state["document_model"],  # Rollback
                    "coherence_regression_detected": True,
                    "coherence_regression_warning": warning_msg,
                    "needs_more_structure_work": False,  # Proceed to polish
                }
```

**Step 3: Sonnet Comparison** (`_check_coherence_regression()`):

```python
async def _check_coherence_regression(
    original_model: DocumentModel,
    edited_model: DocumentModel,
    topic: str,
    edits_summary: str,
) -> Optional[CoherenceComparisonResult]:
    """Call Sonnet to compare document versions for coherence regression."""

    # Render both documents as markdown for comparison
    original_md = original_model.to_markdown()
    edited_md = edited_model.to_markdown()

    user_prompt = COHERENCE_COMPARISON_USER.format(
        topic=topic,
        original_document=original_md,
        edited_document=edited_md,
        edits_summary=edits_summary,
    )

    result = await get_structured_output(
        output_schema=CoherenceComparisonResult,
        user_prompt=user_prompt,
        system_prompt=COHERENCE_COMPARISON_SYSTEM,
        tier=ModelTier.SONNET,  # Use Sonnet for reliable comparison
        max_tokens=2000,
    )
    return result
```

**Recovery Strategy:**
```
Regression Detected (>5% drop)
    ↓
Sonnet Confirms? (confidence >= 0.6)
    ↓ YES
First Time?
    ↓ YES                           ↓ NO
    Rollback + Retry                Rollback + Give Up
    (iteration NOT incremented)     → Proceed to polish with original
    → Try different edits           → Log warning for user
```

## State Fields for Regression Tracking

```python
# In state.py
baseline_coherence_score: Optional[float]       # From iteration 0
coherence_regression_detected: bool              # Final flag after retry failed
coherence_regression_warning: Optional[str]      # Warning message for output
coherence_regression_retry_used: bool            # Track if retry attempted
```

## Files Modified

**Chunked Cross-Comparison (4b7664d):**
- `workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py` - Add `_batch_score_deepseek_cached()` with chunking

**Coherence Regression Detection (45ce342):**
- `workflows/enhance/editing/state.py` - Add baseline and regression tracking fields
- `workflows/enhance/editing/schemas.py` - Add `CoherenceComparisonResult`
- `workflows/enhance/editing/prompts.py` - Add `COHERENCE_COMPARISON_SYSTEM/USER`
- `workflows/enhance/editing/nodes/analyze_structure.py` - Capture baseline on iteration 0
- `workflows/enhance/editing/nodes/verify_structure.py` - Detection, Sonnet comparison, rollback logic

## Prevention

1. **Use cross-comparison for batch scoring** - Always chunk papers for comparative scoring
2. **Capture baselines before edits** - Track pre-edit quality metrics for regression detection
3. **Use higher-tier models for comparison** - Sonnet for coherence assessment is worth the cost
4. **Allow one retry before giving up** - Self-healing without infinite loops
5. **Preserve original on persistent regression** - Better to keep original than compound degradation

## Related Patterns

- [LLM Caching Warmup Pattern](../../patterns/llm-interaction/llm-caching-warmup-pattern.md) - Cache coordination for batch scoring
- [Multi-Phase Document Editing](../../patterns/langgraph/multi-phase-document-editing.md) - Editing workflow structure
- [Model Tier Optimization](./model-tier-optimization.md) - Tier selection for comparison tasks

## References

- Commit (chunking): 4b7664dffed5f9a033c2216f8978acabc0fea5c4
- Commit (regression): 45ce3429ead7dfeaa871392aeceff0aa06c8e82c
