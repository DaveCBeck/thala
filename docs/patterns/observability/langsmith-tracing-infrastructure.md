---
name: langsmith-tracing-infrastructure
title: "LangSmith Tracing Infrastructure: Workflow Cost Visibility"
date: 2026-01-18
category: observability
applicability:
  - "Multi-workflow systems needing cost visibility per run"
  - "LangGraph applications with nested subgraph invocations"
  - "LLM applications requiring filterable observability by metadata"
  - "Production systems needing workflow-level cost aggregation"
components: [workflow_traceable_decorator, trace_config_propagation, dynamic_metadata, llm_cost_tracking]
complexity: low
verified_in_production: false
deprecated: true
deprecated_date: 2026-01-29
deprecated_reason: "Describes utility functions (workflow_traceable, add_trace_metadata, get_trace_config) that don't exist; actual implementation uses direct @traceable decorator"
superseded_by: "./comprehensive-langsmith-tracing.md"
related_solutions: []
tags: [langsmith, tracing, observability, cost-tracking, workflows, decorators, metadata]
---

> **DEPRECATED**: This documentation describes utility functions that were never implemented.
>
> **Reason:** The described `workflow_traceable()` decorator, `add_trace_metadata()`, and `get_trace_config()` functions do not exist. The actual implementation uses the `@traceable` decorator directly with inline metadata.
> **Date:** 2026-01-29
> **See instead:** [Comprehensive LangSmith Tracing](./comprehensive-langsmith-tracing.md)

# LangSmith Tracing Infrastructure: Workflow Cost Visibility

## Intent

Provide centralized tracing utilities that create a single root trace per workflow run, automatically aggregate LLM costs to parent traces, and enable filtering by dynamic metadata in LangSmith UI.

## Motivation

Without structured tracing, LangSmith shows a flat list of individual LLM calls with no way to:
- See total cost per workflow run at a glance
- Group related operations into a trace hierarchy
- Filter runs by quality tier, language, or topic

**The Problem:**
```
LangSmith UI (without structured tracing):
├── ChatAnthropic call (cost: $0.12)
├── ChatAnthropic call (cost: $0.08)
├── ChatAnthropic call (cost: $0.15)
├── ChatAnthropic call (cost: $0.22)  ← Which workflow do these belong to?
├── ChatAnthropic call (cost: $0.09)
└── ... hundreds more
```

**The Solution:**
```
LangSmith UI (with structured tracing):
├── AcademicLitReview [workflow:lit_review] (total: $2.45)
│   ├── DiffusionEngine (cost: $0.85)
│   ├── Clustering (cost: $0.42)
│   └── Synthesis (cost: $1.18)
├── WebResearch [workflow:web_research] (total: $1.23)
│   └── ...
```

## Applicability

Use this pattern when:
- Running multiple workflows that need individual cost tracking
- Using nested LangGraph subgraphs
- Needing to filter runs by quality tier, language, topic, etc.
- Requiring hierarchical trace visibility in LangSmith

Do NOT use this pattern when:
- Single LLM calls without workflow structure
- No LangSmith integration needed
- Trace overhead is unacceptable (minimal, but not zero)

## Structure

```
┌─────────────────────────────────────────────────────────────┐
│  @workflow_traceable(name="MyWorkflow", workflow_type="x") │
│  async def my_workflow(...):                                │
│      add_trace_metadata({...})                              │
│                                                             │
│      ┌──────────────────────────────────────────────────┐  │
│      │  await subgraph.ainvoke(state,                   │  │
│      │      config=get_trace_config())                  │  │
│      │                                                  │  │
│      │  ┌────────────────────────────────────────────┐ │  │
│      │  │  LLM call (get_llm())                      │ │  │
│      │  │  metadata: ls_provider, ls_model_name      │ │  │
│      │  │  → Cost auto-aggregates to parent          │ │  │
│      │  └────────────────────────────────────────────┘ │  │
│      └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

LangSmith UI:
┌─────────────────────────────────────────────────────────────┐
│  MyWorkflow [workflow:x]                                    │
│  Tags: workflow:x                                           │
│  Metadata: quality_tier=standard, topic="AI in medicine"    │
│  Total Cost: $2.45                                          │
│                                                             │
│  └── subgraph                                               │
│      └── ChatAnthropic (cost: $0.42)                        │
│      └── ChatAnthropic (cost: $0.38)                        │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Create Tracing Utilities Module

```python
# workflows/shared/tracing.py

from functools import wraps
from typing import Any, Callable, TypeVar

from langsmith import get_current_run_tree, traceable

F = TypeVar("F", bound=Callable[..., Any])


def workflow_traceable(name: str, workflow_type: str) -> Callable[[F], F]:
    """Decorator for workflow entry points.

    Creates a root trace for the workflow with consistent naming and tags.
    Enables filtering in LangSmith by workflow type.

    Args:
        name: Display name for the trace (e.g., "AcademicLitReview")
        workflow_type: Workflow identifier for filtering (e.g., "lit_review")
    """
    return traceable(
        run_type="chain",
        name=name,
        tags=[f"workflow:{workflow_type}"],
    )


def get_trace_config() -> dict[str, Any]:
    """Get config dict for subgraph invocations that preserves parent trace.

    When invoking a LangGraph subgraph from within a @traceable function,
    call this to get a config dict that links the subgraph trace as a child.

    Returns:
        Config dict with callbacks for parent linking, or empty dict if
        no current trace exists.
    """
    config: dict[str, Any] = {}
    if run_tree := get_current_run_tree():
        config["callbacks"] = run_tree.get_child_callbacks()
    return config


def merge_trace_config(existing_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge trace config with an existing config dict.

    Use when you have an existing config (e.g., with recursion_limit)
    and want to add trace parent linking.
    """
    trace_config = get_trace_config()
    if existing_config is None:
        return trace_config

    merged = dict(existing_config)
    if "callbacks" in trace_config:
        if "callbacks" in merged and merged["callbacks"]:
            if isinstance(merged["callbacks"], list):
                merged["callbacks"] = merged["callbacks"] + trace_config["callbacks"]
            else:
                merged["callbacks"] = [merged["callbacks"]] + trace_config["callbacks"]
        else:
            merged["callbacks"] = trace_config["callbacks"]
    return merged


def add_trace_metadata(metadata: dict[str, Any]) -> None:
    """Add dynamic metadata to the current trace for filtering in LangSmith.

    Call this within a @traceable function to add runtime metadata.
    """
    if run_tree := get_current_run_tree():
        run_tree.add_metadata(metadata)
```

### Step 2: Add LLM Metadata for Cost Tracking

```python
# workflows/shared/llm_utils/models.py

from langchain_anthropic import ChatAnthropic


def get_llm(
    tier: ModelTier,
    max_tokens: int = 4096,
    **kwargs,
) -> ChatAnthropic:
    """Get configured LLM instance with LangSmith cost tracking metadata."""
    return ChatAnthropic(
        model=tier.value,
        max_tokens=max_tokens,
        metadata={
            "ls_provider": "anthropic",
            "ls_model_name": tier.value,  # e.g., "claude-sonnet-4-5-20250929"
        },
        **kwargs,
    )
```

**Important:** LangSmith uses `ls_provider` + `ls_model_name` to:
1. Match against its pricing database
2. Compute costs from token counts
3. Aggregate costs to parent traces automatically

### Step 3: Decorate Workflow Entry Points

```python
# workflows/research/academic_lit_review/graph/api.py

from workflows.shared.tracing import (
    workflow_traceable,
    add_trace_metadata,
    merge_trace_config,
)


@workflow_traceable(name="AcademicLitReview", workflow_type="lit_review")
async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: str = "standard",
    language: str | None = None,
) -> dict:
    """Run academic literature review workflow."""
    # Add dynamic metadata for filtering
    add_trace_metadata({
        "quality_tier": quality,
        "topic": topic[:50],  # Truncate for filtering
        "language": language or "en",
    })

    # Build initial state
    initial_state = {...}

    # Invoke graph with trace config for parent linking
    result = await graph.ainvoke(
        initial_state,
        config=merge_trace_config({
            "recursion_limit": 100,
            "run_name": f"lit_review:{topic[:30]}",
        }),
    )

    return result
```

### Step 4: Propagate Trace to Subgraphs

```python
# workflows/research/academic_lit_review/diffusion_engine/api.py

from workflows.shared.tracing import workflow_traceable, get_trace_config


@workflow_traceable(name="DiffusionEngine", workflow_type="diffusion_engine")
async def run_diffusion_engine(
    topic: str,
    seed_dois: list[str],
    quality_settings: dict,
) -> dict:
    """Run citation diffusion subgraph."""
    initial_state = {...}

    # Use get_trace_config() to link as child of parent workflow
    result = await diffusion_graph.ainvoke(
        initial_state,
        config=get_trace_config(),
    )

    return result
```

## Complete Example

```python
from workflows.shared.tracing import (
    workflow_traceable,
    add_trace_metadata,
    get_trace_config,
    merge_trace_config,
)
from workflows.shared.llm_utils import get_llm, ModelTier


@workflow_traceable(name="MyWorkflow", workflow_type="my_workflow")
async def my_workflow(
    topic: str,
    quality: str = "standard",
) -> dict:
    # Add filterable metadata
    add_trace_metadata({
        "quality_tier": quality,
        "topic": topic[:50],
    })

    # Direct LLM call - cost auto-tracked
    llm = get_llm(ModelTier.SONNET)
    response = await llm.ainvoke([{"role": "user", "content": "..."}])

    # Subgraph invocation - linked as child
    subgraph_result = await my_subgraph.ainvoke(
        {"topic": topic},
        config=get_trace_config(),
    )

    # Graph with existing config
    graph_result = await my_graph.ainvoke(
        {...},
        config=merge_trace_config({
            "recursion_limit": 100,
            "run_name": f"my_graph:{topic[:30]}",
        }),
    )

    return {"result": "..."}


# In LangSmith UI:
# - Filter: tags = "workflow:my_workflow"
# - Filter: metadata_key = "quality_tier" AND metadata_value = "standard"
# - See total cost aggregated from all nested LLM calls
```

## Consequences

### Benefits

- **At-a-glance costs**: Total cost per workflow run visible in LangSmith
- **Full hierarchy**: Drill down into complete trace tree
- **Filterable**: Query by workflow type, quality tier, language, topic
- **Auto-aggregation**: LLM costs roll up to parent traces automatically
- **Minimal overhead**: Just decorator + metadata, no complex setup

### Trade-offs

- **LangSmith dependency**: Requires LangSmith integration
- **Trace overhead**: Small latency added by tracing (negligible in practice)
- **Config propagation**: Must remember to pass `get_trace_config()` to subgraphs

### Alternatives

- **Manual logging**: Log costs at each call (no aggregation)
- **Custom metrics**: Build your own cost tracking (more work)
- **No tracing**: Accept flat list of LLM calls (poor visibility)

## Related Patterns

- [LangSmith Trace Identification](../langgraph/langsmith-trace-identification.md) - Semantic `run_name` for instant identification
- [Conditional Development Tracing](../llm-interaction/conditional-development-tracing.md) - Dev/prod tracing toggle
- [Centralized Logging Configuration](./centralized-logging-configuration.md) - Logging infrastructure
- [Batch API Cost Optimization](../llm-interaction/batch-api-cost-optimization.md) - Cost reduction patterns

## Known Uses in Thala

- `workflows/shared/tracing.py` - Core utilities
- `workflows/research/academic_lit_review/graph/api.py` - Lit review entry point
- `workflows/research/web_research/graph/api.py` - Web research entry point
- `workflows/research/book_finding/graph/api.py` - Book finding entry point
- `workflows/enhance/__init__.py` - Enhancement workflow
- `workflows/wrappers/synthesis/graph/api.py` - Synthesis workflow
- All 18+ workflow entry points decorated with `@workflow_traceable`

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Cost Tracking](https://docs.smith.langchain.com/how_to_guides/cost_tracking)
- [LangSmith Python SDK](https://python.langchain.com/docs/langsmith)
