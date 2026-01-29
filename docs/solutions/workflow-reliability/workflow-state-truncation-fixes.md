---
module: workflows/wrappers/synthesis, workflows/research/academic_lit_review
date: 2026-01-28
problem_type: data_loss
component: integration_nodes, state_loading
symptoms:
  - "Document truncation: sections 8+ and conclusions lost during integration"
  - "0 papers discovered, 0 books selected despite successful workflows"
  - "Large documents (~12K words) missing content after LLM processing"
  - "State not propagating between workflow phases"
root_cause: llm_output_limits, missing_state_retrieval
resolution_type: code_fix
severity: high
tags: [truncation, state-loading, workflow-state-store, programmatic-assembly, llm-limits, data-loss]
---

# Workflow State and Truncation Fixes

## Problem

Two related issues caused data loss in multi-phase workflows:

1. **Document Truncation**: Asking LLMs to re-output entire ~12K word documents caused sections to be lost due to practical token/generation limits
2. **Missing State**: Workflows returning minimal results instead of loading full state from the workflow state store, causing "0 papers" and "0 books" despite successful upstream execution

### Environment

- **Module**: `workflows/wrappers/synthesis`, `workflows/research/academic_lit_review`
- **Python**: 3.12
- **Affected files**:
  - `workflows/research/academic_lit_review/synthesis/nodes/integration_nodes.py`
  - `workflows/wrappers/synthesis/nodes/lit_review.py`
  - `workflows/wrappers/synthesis/nodes/research_workers.py`

### Symptoms

**Truncation Symptom:**
```
Input: 12,800 word document (8 thematic sections + intro + methodology + discussion + conclusions)
Output: Truncated at section 8, conclusions missing

LLM Task: "Take this document, add transitions, generate abstract, output complete document"
Result: Sections 8+, discussion, and conclusions silently dropped
```

**State Loading Symptom:**
```
[INFO] Lit review complete: claude-sonnet-4-5-20250929 (success)
[INFO] Book finding complete: claude-sonnet-4-5-20250929 (success)
[INFO] Synthesis stats: 0 papers discovered, 0 books selected  ← Data lost!

Expected: 127 papers discovered, 8 books selected
```

## Investigation

### What Didn't Work

1. **Increasing max_tokens to 64000**
   - Why it failed: Even with a large token budget, practical generation limits and lossy compression still caused truncation. The model was attempting to re-emit entire documents while maintaining citation integrity.

2. **Adding explicit "preserve all content" instructions**
   - Why it failed: The fundamental issue wasn't instruction following—it was asking the LLM to perform a task that inherently risks content loss (re-outputting massive documents).

3. **Assuming return values contained full state**
   - Why it failed: Return values were intentionally minimal to reduce memory overhead. Full state was saved to the workflow state store but never loaded by downstream phases.

## Root Cause

### Issue 1: LLM-Based Document Assembly

The integration step asked Claude Opus to:
1. Receive ~12K words of input (all sections)
2. Add smooth transitions between sections
3. Generate an abstract
4. Re-output the complete document

This is fundamentally problematic because:
- **Lossy compression**: The model summarizes/rewrites rather than preserves
- **Generation limits**: Even with 64K tokens, generating 12K+ coherent words is unreliable
- **Citation preservation**: Must maintain all `[@KEY]` citations exactly during re-output

```python
# PROBLEMATIC PATTERN
llm = get_llm(tier=ModelTier.OPUS, max_tokens=64000)
response = await invoke_with_cache(
    llm,
    system_prompt=integration_system,  # "Add transitions, output complete document"
    user_prompt=all_sections_combined,  # 12K words
)
integrated = response.content  # Truncated output
```

### Issue 2: Missing State Retrieval

Workflow nodes returned minimal results without loading full state:

```python
# PROBLEMATIC PATTERN in lit_review.py
return {
    "lit_review_result": result,
    # Note: paper_corpus would be retrieved from state store...
    "current_phase": "supervision",
}
# paper_corpus, paper_summaries, zotero_keys all MISSING
```

```python
# PROBLEMATIC PATTERN in research_workers.py
return {
    "book_finding_results": [{
        "processed_books": [],  # Hardcoded empty!
        "zotero_keys": zotero_keys,
    }]
}
```

The workflow state store (`~/.thala/workflow_states/{workflow}/{run_id}.json`) contained all data, but `load_workflow_state()` was never called.

## Solution

### Fix 1: Programmatic Assembly (No LLM for Document Re-output)

Replace LLM-based integration with two-step approach:

**Step 1**: Use LLM only for abstract generation (~250 words)
**Step 2**: Programmatically assemble all sections

```python
# FIXED: integration_nodes.py

def _assemble_document(
    topic: str,
    introduction: str,
    methodology: str,
    thematic_sections: dict[str, str],
    discussion: str,
    conclusions: str,
    clusters: list[dict],
    abstract: str = "",
) -> str:
    """Programmatically assemble the literature review document.

    Combines all sections with proper markdown headers. No LLM needed.
    """
    parts = [f"# Literature Review: {topic}\n"]

    if abstract:
        parts.append(f"## Abstract\n\n{abstract}\n")

    parts.append(f"## 1. Introduction\n\n{introduction}\n")
    parts.append(f"## 2. Methodology\n\n{methodology}\n")

    # Add thematic sections in cluster order
    cluster_order = [c["label"] for c in clusters]
    for i, label in enumerate(cluster_order):
        section_num = i + 3
        section_text = thematic_sections.get(
            label, f"[Section for {label} not available]"
        )
        parts.append(f"## {section_num}. {label}\n\n{section_text}\n")

    # Discussion and conclusions
    discussion_num = len(cluster_order) + 3
    conclusions_num = discussion_num + 1

    parts.append(f"## {discussion_num}. Discussion\n\n{discussion}\n")
    parts.append(f"## {conclusions_num}. Conclusions\n\n{conclusions}\n")

    return "\n".join(parts)


async def integrate_sections_node(state: SynthesisState) -> dict[str, Any]:
    # STEP 1: LLM generates ONLY the abstract (~250 words)
    abstract_target = int(target_words * SECTION_PROPORTIONS["abstract"])

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=1000)  # Small, focused

    response = await invoke_with_cache(
        llm,
        system_prompt=get_abstract_system_prompt(abstract_target),
        user_prompt=ABSTRACT_USER_TEMPLATE.format(
            topic=topic,
            introduction=introduction,  # Context only
            conclusions=conclusions,     # Context only
        ),
    )
    abstract = response.content

    # STEP 2: Programmatic assembly (no LLM)
    integrated = _assemble_document(
        topic=topic,
        introduction=introduction,
        methodology=methodology,
        thematic_sections=thematic_sections,
        discussion=discussion,
        conclusions=conclusions,
        clusters=clusters,
        abstract=abstract,
    )

    return {"integrated_review": integrated}
```

**Key changes:**
| Aspect | Before | After |
|--------|--------|-------|
| Model for integration | Opus (64K tokens) | Sonnet (1K tokens) |
| LLM output size | 12K words | 250 words (abstract only) |
| Section preservation | LLM re-outputs (lossy) | Programmatic (exact) |
| Truncation risk | High | Zero |
| Cost | ~$2.00 | ~$0.02 |

### Fix 2: Load State from Workflow State Store

```python
# FIXED: lit_review.py

from workflows.shared.workflow_state_store import load_workflow_state

async def lit_review_node(state: dict) -> dict[str, Any]:
    # Run the workflow
    if multi_lang_config is not None:
        result = await multi_lang_research(...)
        workflow_name = "multi_lang"
    else:
        result = await academic_lit_review(...)
        workflow_name = "academic_lit_review"

    # CRITICAL: Load full state from store
    paper_corpus = {}
    paper_summaries = {}
    zotero_keys = {}

    run_id = result.get("langsmith_run_id")
    if run_id:
        full_state = load_workflow_state(workflow_name, run_id)
        if full_state:
            paper_corpus = full_state.get("paper_corpus", {})
            paper_summaries = full_state.get("paper_summaries", {})
            zotero_keys = full_state.get("zotero_keys", {})
            logger.info(
                f"Loaded state: {len(paper_corpus)} papers, "
                f"{len(paper_summaries)} summaries, {len(zotero_keys)} zotero keys"
            )

    # Return loaded data for downstream phases
    return {
        "lit_review_result": result,
        "paper_corpus": paper_corpus,
        "paper_summaries": paper_summaries,
        "zotero_keys": zotero_keys,
        "current_phase": "supervision",
    }
```

```python
# FIXED: research_workers.py (book_finding_worker)

async def book_finding_worker(state: dict) -> dict[str, Any]:
    # Run the workflow
    result = await book_finding(...)
    workflow_name = "book_finding"

    # CRITICAL: Load full state from store
    processed_books = []
    zotero_keys = []

    run_id = result.get("langsmith_run_id")
    if run_id:
        full_state = load_workflow_state(workflow_name, run_id)
        if full_state:
            processed_books = full_state.get("processed_books", [])
            zotero_keys = [
                b["zotero_key"] for b in processed_books if b.get("zotero_key")
            ]
            logger.info(
                f"Book worker: loaded {len(processed_books)} books, "
                f"{len(zotero_keys)} zotero keys"
            )

    return {
        "book_finding_results": [{
            "processed_books": processed_books,  # Now populated!
            "zotero_keys": zotero_keys,
        }]
    }
```

### Files Modified

- `workflows/research/academic_lit_review/synthesis/nodes/integration_nodes.py`: Replace LLM-based integration with programmatic assembly
- `workflows/research/academic_lit_review/synthesis/prompts.py`: Remove integration prompt, add abstract-only prompt
- `workflows/wrappers/synthesis/nodes/lit_review.py`: Add `load_workflow_state()` call
- `workflows/wrappers/synthesis/nodes/research_workers.py`: Add `load_workflow_state()` call for book finding

## Prevention

### How to Avoid This

1. **Never ask LLMs to re-output large documents**
   - LLMs are for generation, not copying
   - Use programmatic assembly for document construction
   - LLMs should only generate new content (abstracts, transitions as separate strings)

2. **Always check if state is in the workflow state store**
   - Return values may be intentionally minimal
   - Use `load_workflow_state(workflow_name, run_id)` to get full state
   - Check for `langsmith_run_id` in results to access stored state

3. **Log loaded state quantities**
   ```python
   logger.info(f"Loaded state: {len(paper_corpus)} papers")
   ```
   - Makes data loss visible immediately in logs
   - "0 papers" vs "No persisted state found" are different issues

4. **Use the right model for the task**
   - Abstract generation: Sonnet (1K tokens) is sufficient
   - Full document re-output: Never do this with LLMs

### Test Case

```python
def test_integration_preserves_all_sections():
    """Ensure programmatic assembly doesn't lose sections."""
    thematic_sections = {
        f"Theme {i}": f"Content for theme {i}..." for i in range(8)
    }

    result = _assemble_document(
        topic="Test Topic",
        introduction="Intro content",
        methodology="Method content",
        thematic_sections=thematic_sections,
        discussion="Discussion content",
        conclusions="Conclusions content",
        clusters=[{"label": f"Theme {i}"} for i in range(8)],
        abstract="Abstract content",
    )

    # All sections must be present
    assert "## Abstract" in result
    assert "## 1. Introduction" in result
    assert "## 8. Theme 7" in result  # Last thematic section
    assert "## 10. Conclusions" in result  # Final section
    assert "Conclusions content" in result  # Content preserved
```

## Related

- [Large Document Processing](../api-integration-issues/large-document-processing.md) - Chunking for 40K character limits
- [Long Text Embedding Chunking](../llm-issues/long-text-embedding-chunking.md) - Token limit handling for embeddings
- [Streaming Async Results Pipeline](../async-issues/streaming-async-results-pipeline.md) - Pipeline data flow patterns
- [Multi-Signal Completeness and Retry Logic](./multi-signal-completeness-and-retry-logic.md) - State tracking reliability
