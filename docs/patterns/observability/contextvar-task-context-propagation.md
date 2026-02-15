---
name: contextvar-task-context-propagation
title: "ContextVar-Based Task Context Propagation for LangSmith Tracing"
date: 2026-02-15
category: observability
applicability:
  - "Task queue systems dispatching multiple workflow types"
  - "LangGraph workflows needing consistent trace tags/metadata"
  - "Non-LangChain API calls (Google GenAI, custom HTTP) requiring tracing"
  - "Parallel async executors where task identity must not leak between tasks"
components: [task_context, workflow_executor, entry_points, genai_tracing]
complexity: low
verified_in_production: false
related_solutions:
  - "docs/solutions/async-issues/batch-group-race-condition-contextvars.md"
related_patterns:
  - "docs/patterns/observability/comprehensive-langsmith-tracing.md"
  - "docs/patterns/observability/langsmith-tracing-infrastructure.md"
  - "docs/patterns/langgraph/langsmith-trace-identification.md"
tags: [langsmith, tracing, contextvars, asyncio, task-queue, observability]
---

# ContextVar-Based Task Context Propagation for LangSmith Tracing

## Intent

Propagate task-level identity (task_id, task_type, topic, quality_tier) from a central task executor into all nested workflow code via `ContextVar`, providing consistent LangSmith tags and metadata without threading parameters through every function signature.

## Motivation

Before this pattern, LangSmith tracing was inconsistent across workflows:

- Each workflow entry point manually assembled its own tags/metadata dicts with ad-hoc keys, leading to drift between workflows.
- Non-LangChain API calls (Google Imagen, Gemini) had no tracing — invisible in LangSmith.
- No shared mechanism propagated task identity from the task queue executor into nested workflow code.

**Before (ad-hoc, inconsistent):**
```python
# Each workflow did its own thing
await graph.ainvoke(state, config={
    "tags": ["lit_review"],  # missing task_id, quality tier
    "metadata": {"topic": topic},  # missing task_type, inconsistent keys
})
```

**After (uniform, automatic):**
```python
await graph.ainvoke(state, config={
    "tags": ["workflow:lit_review", *get_trace_tags()],
    "metadata": {**get_trace_metadata(), "topic": topic[:100]},
})
```

## Implementation

### 1. TaskContext dataclass + ContextVar

```python
# core/task_queue/task_context.py
@dataclass(frozen=True)
class TaskContext:
    task_id: str
    task_type: str
    topic: str
    quality_tier: str

_task_context_var: ContextVar[TaskContext | None] = ContextVar(
    "task_context", default=None
)

def get_trace_metadata() -> dict[str, str]:
    ctx = _task_context_var.get()
    if ctx is None:
        return {}
    return {
        "task_id": ctx.task_id, "task_type": ctx.task_type,
        "topic": ctx.topic[:100], "quality_tier": ctx.quality_tier,
    }

def get_trace_tags() -> list[str]:
    ctx = _task_context_var.get()
    if ctx is None:
        return []
    return [f"task:{ctx.task_id}", f"type:{ctx.task_type}"]
```

### 2. Executor sets/clears context

```python
# core/task_queue/workflow_executor.py
set_task_context(
    task_id=task_id, task_type=task_type,
    topic=task_identifier, quality_tier=task.get("quality", "standard"),
)
try:
    result = await workflow.run(task, checkpoint_callback, ...)
finally:
    clear_task_context()  # MUST be in finally to prevent leaks
```

### 3. Workflow entry points consume uniformly

```python
await graph.ainvoke(state, config={
    "tags": ["workflow:web_research", f"quality:{quality}", *get_trace_tags()],
    "metadata": {**get_trace_metadata(), "topic": query[:100]},
})
```

### 4. Non-LangChain calls traced via @traceable + run_tree

```python
# workflows/shared/image_utils.py
@traceable(run_type="tool", name="Imagen_GenerateHeader")
async def generate_article_header(...):
    response = await client.aio.models.generate_images(...)
    rt = get_current_run_tree()
    if rt:
        rt.metadata.update({
            "model": IMAGEN_MODEL,
            "latency_ms": latency_ms,
            "image_count": len(candidates),
        })
```

## Consequences

**Benefits:**
- Single source of truth for task identity — set once, used everywhere
- All LangSmith runs filterable by task_id, task_type, quality_tier
- Non-LangChain calls (Imagen, Gemini) now visible in traces
- No parameter threading — deeply nested code reads ContextVar directly

**Trade-offs:**
- Implicit dependency on ContextVar being set (callers outside task queue get empty dicts, not errors)
- Frozen dataclass prevents mid-task mutations (intentional — forces immutable snapshots)

## Gotchas

1. **Dict spread merge order matters.** `**get_trace_metadata()` last in a dict literal means its `"topic"` key overwrites an earlier explicit one. Truncate inside `get_trace_metadata()` itself to make all consumers safe by default.

2. **`clear_task_context()` must be in `finally`.** If the workflow raises, a stale ContextVar leaks into the next task on the same asyncio task (especially in the parallel executor).

3. **`@traceable` + `get_current_run_tree()`** is the pattern for wrapping non-LangChain SDK calls. The decorator creates the LangSmith run; `get_current_run_tree()` attaches metadata after the call completes (latency, model, result counts).

4. **Frozen dataclass** for `TaskContext` prevents accidental mutation — the ContextVar holds an immutable snapshot per task.

## Known Uses

- `core/task_queue/task_context.py` — the TaskContext definition
- `core/task_queue/workflow_executor.py` — sets/clears context per task dispatch
- All 11 workflow entry points in `workflows/` — consume via `get_trace_tags()` / `get_trace_metadata()`
- `workflows/shared/image_utils.py` — GenAI call tracing
- `workflows/shared/vision_comparison.py` — vision pair comparison tracing

## Related

- [Comprehensive LangSmith Tracing](../observability/comprehensive-langsmith-tracing.md) — broader tracing strategy
- [LangSmith Trace Identification](../langgraph/langsmith-trace-identification.md) — trace ID patterns
- [Batch Group ContextVar Race Condition](../../solutions/async-issues/batch-group-race-condition-contextvars.md) — ContextVar pitfalls in async
