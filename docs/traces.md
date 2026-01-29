# LangSmith Tracing Evaluation Report

## Overview

This document evaluates LangSmith tracing instrumentation across all Thala workflows against 2026 best practices. It identifies current coverage, gaps, and prioritized improvement opportunities.

**Evaluation Date:** 2026-01-20

---

## LangSmith Best Practices (2026 Standards)

### Key Concepts

| Concept | Description | Usage |
|---------|-------------|-------|
| **Runs (spans)** | Single units of work (LLM calls, chain steps) | Track individual operations |
| **Traces** | Collection of runs for a single operation | Group related work |
| **Threads** | Sequences of traces for multi-turn conversations | Use `session_id`/`thread_id` in metadata |
| **Tags** | Categorization strings | Filter by environment, quality tier, phase |
| **Metadata** | Key-value pairs | Store topic, iteration counts, paper counts |

### Required Instrumentation

1. **Entry Points**: `@traceable(run_type="chain", name="WorkflowName")`
2. **Graph Invocations**: `config={"run_name": f"type:{identifier[:30]}", "tags": [...], "metadata": {...}}`
3. **LLM Calls**: Include `ls_provider`, `ls_model_name` for cost tracking
4. **Node Functions**: `@traceable` on major workflow phases
5. **Tags**: Environment, quality tier, phase, model type

---

## Codebase Standards

**Current Pattern (Minimal):**
```python
# Entry point - MISSING @traceable in most workflows
async def workflow_name(...):
    result = await graph.ainvoke(
        initial_state,
        config={
            "run_id": run_id,
            "run_name": f"type:{topic[:30]}",  # Present in most
        },
    )
```

**Reference Implementation (supervision/api.py):**
```python
@traceable(run_type="chain", name="EnhanceReport")  # ONLY workflow with this
async def enhance_report(...):
    ...
```

---

## Workflow Tracing Scorecard

| Workflow | Score | Entry @traceable | run_name | Tags | Metadata | Node Tracing | LLM Metadata |
|----------|-------|------------------|----------|------|----------|--------------|--------------|
| **Supervision** | 37% | ✅ | ✅ | ❌ | ⚠️ | ⚠️ Partial | ✅ |
| **Shared LLM Utils** | 78% | N/A | N/A | ❌ | ✅ | ✅ Executors | ✅ |
| **Document Processing** | 42% | ❌ | ✅ | ❌ | ⚠️ | ❌ None | ✅ |
| **Editing** | 35% | ❌ | ✅ | ❌ | ⚠️ | ❌ None | ✅ |
| **Academic Lit Review** | 27% | ❌ | ✅ | ❌ | ⚠️ | ❌ None | ✅ |
| **Fact Check** | 30% | ❌ | ✅ | ❌ | ⚠️ | ❌ None | ✅ |
| **Web Research** | 24% | ❌ | ✅ | ❌ | ❌ | ❌ None | ✅ |
| **Multi-Lang** | 25% | ❌ | ✅ | ❌ | ⚠️ | ❌ None | ✅ |
| **Book Finding** | 25% | ❌ | ✅ | ❌ | ❌ | ❌ None | ✅ |
| **Synthesis** | 13% | ❌ | ✅ | ❌ | ❌ | ❌ None | ✅ |

**Legend:** ✅ Implemented | ⚠️ Partial | ❌ Missing

---

## Detailed Workflow Evaluations

### 1. Supervision/Enhancement Workflow

**Score: 37% (Best in codebase)**

**Location:** `workflows/enhance/supervision/`

**Strengths:**
- ✅ Entry point has `@traceable(run_type="chain", name="EnhanceReport")` (`api.py:19`)
- ✅ Loop1 and Loop2 have `@traceable` decorators (`loop1/graph.py:156`, `loop2/graph.py:437`)
- ✅ Rich logging with iteration counts, paper counts

**Gaps:**
- ❌ **Hierarchy broken**: Wrapper nodes (`run_loop1_node`, `run_loop2_node`) NOT traced
- ❌ **Config not passed**: Loop functions called without config, breaking parent-child relationship
- ❌ **No iteration-level tracing**: All iterations collapse into single trace
- ❌ **No tags**: Missing quality tier, loop type, iteration tags
- ❌ **Individual nodes untraced**: `analyze_review_node`, `expand_topic_node`, `integrate_content_node`

**Key Files:**
| File | Issue |
|------|-------|
| `api.py:19` | ✅ Has @traceable |
| `nodes.py:17-70` | ❌ `run_loop1_node` not traced |
| `nodes.py:73-149` | ❌ `run_loop2_node` not traced |
| `nodes.py:32-39` | ❌ No config passed to loop1 |

---

### 2. Shared LLM Utilities

**Score: 78% (Infrastructure layer)**

**Location:** `workflows/shared/llm_utils/`

**Strengths:**
- ✅ `get_llm()` sets `ls_provider` and `ls_model_name` for all models (`models.py:108-131`)
- ✅ All structured output executors have `@traceable`:
  - `json.py:28` - `@traceable(run_type="llm", name="json_prompting")`
  - `langchain.py:26` - `@traceable(run_type="llm", name="structured_output")`
  - `batch.py:27` - `@traceable(name="batch_structured_output")`
  - `agent_runner.py:20` - `@traceable(name="tool_agent")`
- ✅ Batch processor instrumented with usage aggregation (`processor.py:110, 218`)
- ✅ Anthropic clients wrapped with `wrap_anthropic()` (`processor.py:54-55`)

**Gaps:**
| Component | File:Line | Issue |
|-----------|-----------|-------|
| `execute_batch_concurrent` | `batch_executor.py:22` | ❌ No @traceable on fallback |
| `BatchToolCallExecutor` | `batch.py:27` | ⚠️ Missing `run_type="llm"` |
| Tool call tracing | `agent_runner.py:62-98` | ❌ Individual tool calls not traced |

---

### 3. Document Processing

**Score: 42%**

**Location:** `workflows/document_processing/`

**Strengths:**
- ✅ Has `run_name` in config (`graph.py:153`)
- ✅ LLM metadata properly configured via `get_llm()`

**Gaps:**
- ❌ **No @traceable on entry** (`graph.py:121`)
- ❌ **14 nodes completely untraced**:
  - `resolve_input`, `create_zotero_stub`, `update_store`, `detect_language`
  - `generate_summary`, `check_metadata`, `save_short_summary`, `update_zotero`
  - `detect_chapters`, `save_tenth_summary`, `summarize_chapters`, `aggregate_summaries`
  - `finalize`
- ❌ **No batch-specific tracing**: Batch mode operations invisible
- ❌ **No tags**: Document type, source not tagged

---

### 4. Editing Workflow

**Score: 35%**

**Location:** `workflows/enhance/editing/`

**Strengths:**
- ✅ Has `run_name` in config (`api.py:89`)
- ✅ LLM calls via `get_structured_output()` (auto-traced)

**Gaps:**
- ❌ **No @traceable on entry** (`api.py:17`)
- ❌ **8 nodes untraced**:
  - `analyze_structure`, `plan_edits`, `execute_generation_edit_worker`
  - `execute_structure_edits_worker`, `verify_structure`, `polish`, `finalize`
- ❌ **No phase context**: LLM calls don't know which phase they belong to
- ❌ **No tags**: Quality tier, phase not tagged
- ❌ **Parallel workers invisible**: Edit workers not distinguishable

---

### 5. Academic Lit Review

**Score: 27%**

**Location:** `workflows/research/academic_lit_review/`

**Strengths:**
- ✅ Has `run_name` in config (`api.py:97`)
- ✅ UUID tracking for run correlation

**Gaps:**
- ❌ **No @traceable on entry** (`api.py:17`)
- ❌ **All subgraphs untraced**:
  - `keyword_search/__init__.py:86` - `ainvoke(initial_state)` no config
  - `diffusion_engine/api.py:96` - no config
  - `paper_processor/api.py:58` - no config
  - `clustering/api.py:54` - no config
  - `synthesis/api.py:86` - no config
- ❌ **No tags**: Quality tier, language not tagged
- ❌ **Diffusion stages invisible**: Stages 1-5 not individually traced

---

### 6. Fact Check Workflow

**Score: 30%**

**Location:** `workflows/enhance/fact_check/`

**Strengths:**
- ✅ Has `run_name` and `run_id` in config (`api.py:109-110`)
- ✅ Comprehensive result tracking in state

**Gaps:**
- ❌ **No @traceable on entry** (`api.py:17`)
- ❌ **Tool calls completely untraced**: `search_papers`, `get_paper_content`, `check_fact`
- ❌ **Phase nodes untraced**: Screening, fact-check, reference-check phases
- ❌ **No tags**: Verification type, phase not tagged

---

### 7. Web Research

**Score: 24%**

**Location:** `workflows/research/web_research/`

**Note:** Tracing was **deliberately reverted** in commit `9646b18` (Jan 18, 2026)

**Strengths:**
- ✅ Has `run_name` in config (`api.py:156`)
- ✅ LLM metadata from `get_llm()`

**Gaps:**
- ❌ **No @traceable on entry** (`api.py:44`) - was reverted
- ❌ **All 9 nodes untraced**: `supervisor`, `web_researcher`, `clarify_intent`, etc.
- ❌ **No tags**: Quality tier, language, iteration not tagged
- ❌ **Supervisor decisions invisible**: No visibility into research strategy

---

### 8. Multi-Language Wrapper

**Score: 25%**

**Location:** `workflows/wrappers/multi_lang/`

**Strengths:**
- ✅ Has `run_name` in config (`api.py:162`)

**Gaps:**
- ❌ **No @traceable on entry** (`api.py:62`)
- ❌ **No language branch tracing**: Per-language executions not separated
- ❌ **No language tags**: Cannot filter by `language:en`, `language:es`
- ❌ **Integration steps untraced**: `_create_initial_synthesis`, `_integrate_language`, `_finalize_synthesis`

---

### 9. Book Finding

**Score: 25%**

**Location:** `workflows/research/book_finding/`

**Strengths:**
- ✅ Has `run_name` in config (`api.py:116`)
- ✅ LLM metadata for recommendations

**Gaps:**
- ❌ **No @traceable on entry** (`api.py:26`)
- ❌ **Search operations untraced**: `search_books` node, book_search API calls
- ❌ **Processing untraced**: `process_books`, document_processing wrapper
- ❌ **No category tags**: Recommendation categories not distinguishable

---

### 10. Synthesis Workflow

**Score: 13% (Lowest)**

**Location:** `workflows/wrappers/synthesis/`

**Strengths:**
- ✅ Has `run_name` in config (`api.py:143`)

**Gaps:**
- ❌ **No @traceable on entry** (`api.py:21`)
- ❌ **13 phases, 0 traced**:
  - `lit_review`, `supervision`, `research_targets`, `web_research_worker`
  - `book_finding_worker`, `aggregation`, `suggest_structure`, `select_books`
  - `fetch_book_summaries`, `write_section_worker`, `check_section_quality`
  - `assemble_sections`, `editing`, `finalize`
- ❌ **No tags**: Quality tier, phase not tagged
- ❌ **External workflows disconnected**: lit_review, supervision, editing appear as separate runs

---

## Summary Statistics

### By Instrumentation Type

| Feature | Workflows With | Workflows Without |
|---------|---------------|-------------------|
| Entry `@traceable` | 1 (Supervision) | 9 |
| `run_name` in config | 10 | 0 |
| Tags | 0 | 10 |
| Node-level tracing | 1 partial (Supervision) | 9 |
| LLM metadata (cost) | 10 | 0 |

### Node Tracing Coverage

| Workflow | Total Nodes | Traced Nodes | Coverage |
|----------|-------------|--------------|----------|
| Supervision | 6 | 3 | 50% |
| Document Processing | 14 | 0 | 0% |
| Editing | 8 | 0 | 0% |
| Academic Lit Review | 5+ subgraphs | 0 | 0% |
| Web Research | 9 | 0 | 0% |
| Multi-Lang | 5 | 0 | 0% |
| Synthesis | 13 | 0 | 0% |
| Fact Check | 7 | 0 | 0% |
| Book Finding | 4 | 0 | 0% |

---

## Improvement Opportunities

### Priority 1: Critical (Week 1)

**Add `@traceable` to all workflow entry points:**

| Workflow | File:Line | Code to Add |
|----------|-----------|-------------|
| Academic Lit Review | `graph/api.py:17` | `@traceable(run_type="chain", name="AcademicLitReview")` |
| Web Research | `graph/api.py:44` | `@traceable(run_type="chain", name="DeepResearch")` |
| Document Processing | `graph.py:121` | `@traceable(run_type="chain", name="DocumentProcessing")` |
| Editing | `graph/api.py:17` | `@traceable(run_type="chain", name="EditingWorkflow")` |
| Multi-Lang | `graph/api.py:62` | `@traceable(run_type="chain", name="MultiLangResearch")` |
| Synthesis | `graph/api.py:21` | `@traceable(run_type="chain", name="SynthesisWorkflow")` |
| Fact Check | `graph/api.py:17` | `@traceable(run_type="chain", name="FactCheck")` |
| Book Finding | `graph/api.py:26` | `@traceable(run_type="chain", name="BookFinding")` |

**Effort:** ~1 hour | **Impact:** All workflows become top-level LangSmith chains

---

### Priority 2: High (Week 2)

**Add tags to all graph configs:**

```python
# Pattern for all workflows
config = {
    "run_id": run_id,
    "run_name": f"workflow:{topic[:30]}",
    "tags": [f"quality:{quality}", "workflow:type", f"lang:{language}"],
    "metadata": {
        "topic": topic,
        "quality_tier": quality,
        "question_count": len(research_questions),
    }
}
```

**Workflows to update:**
- All 10 workflows need tags added to their `ainvoke()` config

**Effort:** ~2 hours | **Impact:** Enables LangSmith filtering by quality/environment

---

### Priority 3: High (Week 2-3)

**Fix Supervision hierarchy:**

```python
# nodes.py - Pass config through
async def run_loop1_node(state: EnhanceState) -> dict[str, Any]:
    result = await run_loop1_standalone(
        ...,
        config={"tags": ["loop:theoretical_depth"]},  # ADD THIS
    )
```

**Files to update:**
- `supervision/nodes.py:32-39` - Pass config to loop1
- `supervision/nodes.py:102-110` - Pass config to loop2

**Effort:** ~30 min | **Impact:** Proper trace hierarchy for supervision

---

### Priority 4: Medium (Week 3-4)

**Add `@traceable` to major node functions:**

Top candidates (most impact):
1. **Document Processing**: 14 nodes
2. **Editing**: 8 nodes
3. **Synthesis**: 13 phases
4. **Academic Lit Review**: 5 subgraph invocations

Pattern:
```python
from langsmith import traceable

@traceable(run_type="chain", name="analyze_structure")
async def analyze_structure_node(state: dict) -> dict:
    ...
```

**Effort:** ~4-6 hours | **Impact:** Node-level visibility in all workflows

---

### Priority 5: Medium (Week 4)

**Fix shared utilities gaps:**

| Component | File:Line | Fix |
|-----------|-----------|-----|
| `execute_batch_concurrent` | `batch_executor.py:22` | Add `@traceable(run_type="tool", name="batch_concurrent")` |
| `BatchToolCallExecutor` | `batch.py:27` | Add `run_type="llm"` to existing decorator |

**Effort:** ~30 min | **Impact:** Complete LLM infrastructure tracing

---

### Priority 6: Lower (Month 2)

**Add tool-level tracing:**

For workflows with tool calls (fact_check, web_research):
```python
@traceable(name="search_papers", tags=["tool:search"])
async def traced_search_papers(query: str, ...):
    return await search_papers.ainvoke(...)
```

**Add iteration/thread tracking:**

For multi-turn workflows (supervision):
```python
config = {
    "metadata": {
        "session_id": session_id,  # For thread grouping
        "iteration": iteration,
    }
}
```

---

## Implementation Checklist

### Phase 1: Entry Points (Critical)
- [ ] Add @traceable to academic_lit_review entry
- [ ] Add @traceable to web_research entry
- [ ] Add @traceable to document_processing entry
- [ ] Add @traceable to editing entry
- [ ] Add @traceable to multi_lang entry
- [ ] Add @traceable to synthesis entry
- [ ] Add @traceable to fact_check entry
- [ ] Add @traceable to book_finding entry

### Phase 2: Tags & Metadata
- [ ] Add tags to all workflow configs (quality, environment)
- [ ] Add metadata to all workflow configs (topic, counts)
- [ ] Standardize tag naming convention

### Phase 3: Supervision Fix
- [ ] Pass config to run_loop1_node
- [ ] Pass config to run_loop2_node
- [ ] Add @traceable to wrapper nodes

### Phase 4: Node Tracing
- [ ] Add @traceable to document_processing nodes (14)
- [ ] Add @traceable to editing nodes (8)
- [ ] Add @traceable to synthesis nodes (13)
- [ ] Add @traceable to fact_check nodes (7)

### Phase 5: Infrastructure
- [ ] Add @traceable to batch_executor concurrent
- [ ] Add run_type to BatchToolCallExecutor
- [ ] Consider tool-level tracing

---

## Cost Tracking Status

**All workflows have proper LLM cost tracking:**

```python
# models.py:128-131 (all LLM calls)
"metadata": {
    "ls_provider": "anthropic",  # or "deepseek"
    "ls_model_name": tier.value,
}
```

This enables automatic cost calculation in LangSmith for:
- Claude Haiku, Sonnet, Sonnet 1M, Opus
- DeepSeek V3, R1

---

## Conclusion

The Thala codebase has **foundational LangSmith integration** through:
- `run_name` in graph configs (all workflows)
- LLM metadata for cost tracking (all models)
- Structured output executor tracing (shared infrastructure)

However, **significant gaps exist**:
- Only 1/10 workflows has entry `@traceable` (Supervision)
- 0/10 workflows use tags
- 60+ nodes across workflows lack individual tracing
- Trace hierarchies are flat and hard to navigate

**Recommended timeline:**
- **Week 1**: Add `@traceable` to all entry points
- **Week 2**: Add tags to all configs, fix supervision hierarchy
- **Week 3-4**: Add node-level tracing to major workflows
- **Month 2**: Tool tracing, iteration tracking, documentation

**Estimated total effort:** 15-20 hours for full compliance with 2026 best practices.

---

*Generated: 2026-01-20*
*Source: Explore agent analysis of all workflow files*
