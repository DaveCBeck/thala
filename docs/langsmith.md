# LangSmith Tracing Guide

This document describes the tracing patterns used in this codebase for LangSmith integration. Following these patterns ensures:

1. **At-a-glance workflow costs** - See total cost per workflow run in LangSmith UI
2. **Full trace visibility** - Drill into complete trace hierarchy for any workflow
3. **Filterable metadata** - Query runs by quality tier, language, workflow type, etc.

## Core Principles

### 1. Single Root Trace per Workflow

Every workflow entry point uses `@workflow_traceable` to create the root trace:

```python
from workflows.shared.tracing import workflow_traceable, add_trace_metadata

@workflow_traceable(name="MyWorkflow", workflow_type="my_workflow")
async def my_workflow(topic: str, quality: str = "standard") -> dict:
    # Add dynamic metadata for filtering
    add_trace_metadata({
        "quality_tier": quality,
        "topic": topic[:50],
    })

    # ... workflow logic
```

This creates:
- One trace = one workflow run in LangSmith UI
- All child operations roll up costs to this root
- Filtering by `workflow:my_workflow` tag

### 2. Automatic Cost Tracking via LLM Metadata

All LLM instances created via `get_llm()` automatically include LangSmith metadata:

```python
# In workflows/shared/llm_utils/models.py
kwargs = {
    "model": tier.value,
    "metadata": {
        "ls_provider": "anthropic",
        "ls_model_name": tier.value,  # e.g., "claude-sonnet-4-5-20250929"
    },
}
```

LangSmith uses `ls_provider` + `ls_model_name` to:
1. Match against its pricing database
2. Compute costs from token counts
3. Aggregate to parent traces automatically

**Important**: Always use `get_llm()` from `workflows.shared.llm_utils` - never instantiate `ChatAnthropic` directly.

### 3. Parent Propagation for Subgraphs

When invoking subgraphs, always pass trace config to maintain the hierarchy:

```python
from workflows.shared.tracing import get_trace_config, merge_trace_config

# Simple case - no existing config
result = await subgraph.ainvoke(state, config=get_trace_config())

# With existing config (e.g., recursion_limit)
result = await graph.ainvoke(
    state,
    config=merge_trace_config({
        "recursion_limit": 100,
        "run_name": f"my_graph:{topic[:30]}",
    }),
)
```

**Without this**: Subgraph traces become orphaned at the root level, breaking the hierarchy and cost aggregation.

### 4. Dynamic Metadata for Filtering

Add runtime metadata for powerful filtering in LangSmith:

```python
from workflows.shared.tracing import add_trace_metadata

@workflow_traceable(name="AcademicLitReview", workflow_type="lit_review")
async def academic_lit_review(topic: str, quality: str, language: str):
    add_trace_metadata({
        "quality_tier": quality,
        "language": language,
        "topic": topic[:50],  # Truncate for filtering
    })
```

This enables LangSmith queries like:
- `metadata_key = "quality_tier" AND metadata_value = "high_quality"`
- `metadata_key = "language" AND metadata_value = "es"`

## API Reference

### `workflow_traceable(name, workflow_type)`

Decorator for workflow entry points. Creates a root trace with consistent naming.

```python
@workflow_traceable(name="BookFinding", workflow_type="book_finding")
async def find_books(...):
    ...
```

- `name`: Display name in LangSmith (e.g., "BookFinding")
- `workflow_type`: Tag suffix for filtering (creates `workflow:book_finding` tag)

### `get_trace_config()`

Returns a config dict for subgraph invocations that links to the current trace.

```python
result = await subgraph.ainvoke(state, config=get_trace_config())
```

Returns empty dict if no current trace exists (safe to call unconditionally).

### `merge_trace_config(existing_config)`

Merges trace callbacks with an existing config dict.

```python
config = merge_trace_config({
    "recursion_limit": 100,
    "run_name": "my_graph",
})
result = await graph.ainvoke(state, config=config)
```

Handles `None` input gracefully.

### `add_trace_metadata(metadata)`

Adds key-value metadata to the current trace for filtering.

```python
add_trace_metadata({
    "quality_tier": "high_quality",
    "language": "es",
    "topic": topic[:50],
})
```

Call this early in the workflow function, after the decorator has created the trace.

## Checklist for New Workflows

When creating a new workflow entry point:

- [ ] Import from `workflows.shared.tracing`
- [ ] Add `@workflow_traceable(name="...", workflow_type="...")` decorator
- [ ] Call `add_trace_metadata()` with relevant parameters (quality, language, topic, etc.)
- [ ] Use `merge_trace_config()` when invoking the main graph
- [ ] Use `get_trace_config()` for any subgraph invocations

When creating internal subgraphs/utilities:

- [ ] Accept `config: dict | None = None` parameter if appropriate
- [ ] Pass `config=get_trace_config()` or `config=merge_trace_config(config)` to graph invocations

## Workflow Types Registry

Current workflow types (for `workflow_type` parameter):

| Workflow Type | Description |
|--------------|-------------|
| `lit_review` | Academic literature review |
| `web_research` | Deep web research |
| `book_finding` | Book recommendations |
| `enhance_full` | Full enhancement (supervision + editing) |
| `enhance_supervision` | Supervision loops only |
| `enhance_editing` | Editing workflow only |
| `synthesis` | Multi-source synthesis |
| `document_processing` | Document ingestion |
| `paper_processor` | Paper processing subgraph |
| `clustering` | Paper clustering |
| `synthesis_mapreduce` | Map-reduce synthesis |
| `iterative_synthesis` | Iterative synthesis |
| `diffusion_engine` | Citation diffusion |
| `keyword_search` | Keyword-based paper discovery |
| `citation_network` | Citation network expansion |
| `supervision_loop1` | Theoretical depth loop |
| `supervision_loop2` | Literature expansion loop |
| `multi_lang` | Multi-language research |

## Troubleshooting

### Orphaned traces at root level

**Symptom**: Subgraph runs appear as separate root traces instead of children.

**Fix**: Add `config=get_trace_config()` to the subgraph invocation.

### Missing cost data

**Symptom**: Traces show token counts but no costs.

**Fix**: Ensure `ls_provider` and `ls_model_name` are set in LLM metadata. Use `get_llm()` instead of direct `ChatAnthropic` instantiation.

### Traces not appearing

**Symptom**: No traces in LangSmith despite code running.

**Fix**: Check that `LANGSMITH_TRACING=true` is set in environment (handled by `core/config.py` in dev mode).
