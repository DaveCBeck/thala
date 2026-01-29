---
name: comprehensive-langsmith-tracing
title: "Comprehensive LangSmith Tracing: Full Workflow Observability"
date: 2026-01-28
category: observability
applicability:
  - "Multi-workflow systems needing complete execution visibility"
  - "LangGraph applications with nested subgraphs and supervision loops"
  - "Production systems requiring filtering by quality tier, workflow type, language"
  - "Test scripts that need complete trace data before exit"
components: [workflow_traceable, node_traceable, tool_wrappers, config_propagation, trace_flush]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [langsmith, tracing, observability, workflows, nodes, tools, testing, flush]
---

# Comprehensive LangSmith Tracing: Full Workflow Observability

## Intent

Implement consistent @traceable decorators across all workflow entry points, node functions, and tool calls with standardized tags/metadata for filtering, plus proper trace flush handling to ensure complete data capture.

## Motivation

Partial tracing coverage creates blind spots in observability:

**The Problem:**
```
LangSmith UI (incomplete tracing):
├── AcademicLitReview                    ← Workflow visible
│   ├── ??? (gap)                        ← Node execution invisible
│   ├── ChatAnthropic call               ← LLM visible (auto)
│   ├── ??? (gap)                        ← Tool calls invisible
│   └── ??? (gap)                        ← Supervision loops invisible

Issues:
- Cannot trace execution path through nodes
- Tool calls not visible for evaluation
- No filtering by quality tier or language
- Test runs show "interrupted" (red stop sign) even when successful
```

**The Solution:**
```
LangSmith UI (comprehensive tracing):
├── AcademicLitReview [workflow:lit_review, quality:standard, lang:en]
│   ├── EditingParseDocument             ← Node visible
│   │   └── ChatAnthropic call           ← LLM call
│   ├── SupervisionLoop1Node             ← Supervision visible
│   │   ├── traced_search_papers         ← Tool visible
│   │   └── run_tool_agent               ← Agent visible
│   ├── batch_structured_output          ← Batch API visible
│   └── EditingFinalize                  ← Final node

Filtering available:
- Tags: quality:standard, workflow:lit_review, lang:en
- Metadata: topic, question_count, quality_tier
- All traces complete (green checkmarks) after test exit
```

## Applicability

Use this pattern when:
- Building multi-workflow systems with LangGraph
- Need filtering by quality tier, workflow type, language in LangSmith
- Running test scripts that need complete trace data
- Requiring visibility into tool calls and supervision loops
- Evaluating workflow performance with complete execution traces

Do NOT use this pattern when:
- Single LLM call with no workflow structure
- No LangSmith integration needed
- Minimal observability requirements

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  Workflow Entry Point                                              │
│  @traceable(run_type="chain", name="WorkflowName")                │
│  config = {                                                        │
│      "run_name": f"workflow:{topic[:30]}",                        │
│      "tags": ["quality:standard", "workflow:lit_review"],          │
│      "metadata": {"topic": topic[:100], "quality_tier": quality}, │
│  }                                                                 │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Node Functions                                                    │
│  @traceable(run_type="chain", name="NodeName")                    │
│                                                                    │
│  ┌─────────────────────┐  ┌─────────────────────┐                 │
│  │ EditingParseDocument│  │ SupervisionLoop1Node│                 │
│  └─────────────────────┘  └─────────────────────┘                 │
│           │                        │                               │
│           ▼                        ▼                               │
│  ┌─────────────────────┐  ┌─────────────────────┐                 │
│  │ LLM Call (auto)     │  │ run_tool_agent      │                 │
│  │ ChatAnthropic       │  │ @traceable          │                 │
│  └─────────────────────┘  └─────────────────────┘                 │
│                                    │                               │
│                                    ▼                               │
│                           ┌─────────────────────┐                 │
│                           │ traced_search_papers│                 │
│                           │ @traceable(tool)    │                 │
│                           └─────────────────────┘                 │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Test Script Exit                                                  │
│  try:                                                              │
│      asyncio.run(main())                                          │
│  finally:                                                          │
│      wait_for_all_tracers()  # Flush before exit                  │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Decorate Workflow Entry Points

```python
# workflows/research/academic_lit_review/graph/api.py

from langsmith import traceable


@traceable(run_type="chain", name="AcademicLitReview")
async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: str = "standard",
    language: str = "en",
) -> dict[str, Any]:
    """Run academic literature review with full tracing."""
    # Build initial state
    initial_state = {...}

    # Invoke with tags and metadata for filtering
    result = await workflow_graph.ainvoke(
        initial_state,
        config={
            "run_name": f"lit_review:{topic[:30]}",
            "tags": [
                f"quality:{quality}",
                "workflow:lit_review",
                f"lang:{language}",
            ],
            "metadata": {
                "topic": topic[:100],  # Truncate for searchability
                "quality_tier": quality,
                "question_count": len(research_questions),
                "language": language,
            },
        },
    )

    return result
```

**Standardized Tags:**
| Tag | Values | Purpose |
|-----|--------|---------|
| `quality:{tier}` | test, quick, standard, comprehensive, high_quality | Filter by quality |
| `workflow:{type}` | lit_review, editing, fact_check, synthesis, web_research | Filter by workflow |
| `lang:{code}` | en, es, de, ja, zh, etc. | Filter by language |

### Step 2: Decorate All Node Functions

```python
# workflows/enhance/editing/nodes/parse_document.py

from langsmith import traceable


@traceable(run_type="chain", name="EditingParseDocument")
async def parse_document_node(state: dict) -> dict[str, Any]:
    """Parse document with tracing."""
    # Node implementation
    ...
```

```python
# workflows/enhance/supervision/nodes.py

from langchain_core.runnables import RunnableConfig
from langsmith import traceable


@traceable(run_type="chain", name="SupervisionLoop1Node")
async def run_loop1_node(
    state: EnhanceState,
    config: RunnableConfig,  # Accept config for hierarchy
) -> dict[str, Any]:
    """Run Loop 1 with proper config hierarchy."""
    result = await run_loop1_standalone(
        # ... parameters ...
        config=config,  # CRITICAL: Pass to maintain trace hierarchy
    )
    return result
```

**Naming Convention:**
- Format: `{Workflow}{Phase}` (PascalCase)
- Examples: `EditingParseDocument`, `FactCheckSectionWorker`, `SupervisionLoop1Node`

### Step 3: Create Tool Wrappers Module

```python
# workflows/shared/tracing/__init__.py

from .tool_wrappers import (
    traced_tool_call,
    traced_search_papers,
    traced_get_paper_content,
    traced_web_search,
    traced_scrape_url,
)

__all__ = [
    "traced_tool_call",
    "traced_search_papers",
    "traced_get_paper_content",
    "traced_web_search",
    "traced_scrape_url",
]
```

```python
# workflows/shared/tracing/tool_wrappers.py

import asyncio
from typing import Any

from langsmith import traceable


@traceable(run_type="tool", name="execute_tool_call")
async def traced_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_func: Any,
) -> Any:
    """Generic traced wrapper for any tool call."""
    if hasattr(tool_func, "ainvoke"):
        result = await tool_func.ainvoke(tool_args)
    elif callable(tool_func):
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**tool_args)
        else:
            result = tool_func(**tool_args)
    return result


@traceable(run_type="tool", name="search_papers", tags=["tool:search"])
async def traced_search_papers(
    query: str,
    max_results: int = 10,
    **kwargs: Any,
) -> Any:
    """Traced wrapper for paper search operations."""
    from langchain_tools import search_papers
    return await search_papers.ainvoke({
        "query": query,
        "max_results": max_results,
        **kwargs,
    })


@traceable(run_type="tool", name="get_paper_content", tags=["tool:retrieval"])
async def traced_get_paper_content(
    paper_id: str,
    **kwargs: Any,
) -> Any:
    """Traced wrapper for paper content retrieval."""
    from langchain_tools import get_paper_content
    return await get_paper_content.ainvoke({
        "paper_id": paper_id,
        **kwargs,
    })


@traceable(run_type="tool", name="web_search", tags=["tool:web"])
async def traced_web_search(
    query: str,
    **kwargs: Any,
) -> Any:
    """Traced wrapper for web search operations."""
    from langchain_tools import web_search
    return await web_search.ainvoke({"query": query, **kwargs})
```

### Step 4: Trace Agent Tool Execution

```python
# workflows/shared/llm_utils/structured/executors/agent_runner.py

from langsmith import traceable


@traceable(run_type="tool", name="execute_tool_call")
async def _execute_tool_call(
    tool: BaseTool,
    tool_name: str,
    tool_args: dict,
) -> str:
    """Execute a single tool call with LangSmith tracing."""
    result = await tool.ainvoke(tool_args)
    return str(result) if result is not None else ""


@traceable(name="tool_agent")
async def run_tool_agent(
    llm: ChatAnthropic,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    output_schema: Type[T],
    max_tool_calls: int = 12,
) -> T:
    """Run agent loop with full tracing."""
    for iteration in range(max_tool_calls):
        response = await llm.ainvoke(messages)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = await _execute_tool_call(
                    tool=tool_map[tool_call["name"]],
                    tool_name=tool_call["name"],
                    tool_args=tool_call["args"],
                )
                # ... continue agent loop
```

### Step 5: Add Trace Flush to Test Scripts

```python
# testing/test_academic_lit_review.py

import asyncio

from langchain_core.tracers.langchain import wait_for_all_tracers


async def main():
    """Run test workflow."""
    result = await academic_lit_review(
        topic="Large Language Models",
        research_questions=["What are recent advances?"],
        quality="standard",
    )
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # CRITICAL: Wait for LangSmith to flush all trace data
        wait_for_all_tracers()
```

**Why this is needed:** LangSmith uses background threads to send trace data. Without `wait_for_all_tracers()`, the main thread exits before traces are flushed, causing runs to appear as "interrupted" (red stop sign) even when they completed successfully.

## Complete Example

```python
# Workflow entry point with full tracing
from langsmith import traceable


@traceable(run_type="chain", name="MyWorkflow")
async def my_workflow(
    topic: str,
    quality: str = "standard",
) -> dict[str, Any]:
    """Workflow with comprehensive tracing."""
    result = await workflow_graph.ainvoke(
        {"topic": topic},
        config={
            "run_name": f"my_workflow:{topic[:30]}",
            "tags": [f"quality:{quality}", "workflow:my_workflow"],
            "metadata": {"topic": topic[:100], "quality_tier": quality},
        },
    )
    return result


# Node with tracing
@traceable(run_type="chain", name="MyWorkflowProcess")
async def process_node(state: dict) -> dict[str, Any]:
    """Process node with tracing."""
    # Use traced tool wrapper
    results = await traced_search_papers(
        query=state["topic"],
        max_results=10,
    )
    return {"results": results}


# Test script with flush
if __name__ == "__main__":
    from langchain_core.tracers.langchain import wait_for_all_tracers

    try:
        asyncio.run(main())
    finally:
        wait_for_all_tracers()
```

### LangSmith Filter Examples

```
# Filter by quality tier
tags = "quality:standard"

# Filter by workflow type
tags = "workflow:lit_review"

# Filter by language
tags = "lang:es"

# Combined filter
tags = "quality:comprehensive" AND tags = "workflow:synthesis"

# Metadata filter
metadata.quality_tier = "high_quality"
metadata.topic CONTAINS "machine learning"
```

## Consequences

### Benefits

- **Complete visibility**: Every workflow, node, and tool call traced
- **Powerful filtering**: Query by quality tier, workflow type, language
- **Trace hierarchy**: Nested execution paths visible
- **Clean test output**: All traces show green checkmarks (not interrupted)
- **Evaluation ready**: Tool calls visible for agent evaluation
- **Cost tracking**: Automatic via LangSmith when properly traced

### Trade-offs

- **Decorator overhead**: Small latency added by tracing
- **Code verbosity**: @traceable on every function
- **Config propagation**: Must pass RunnableConfig through supervision
- **Test cleanup**: Must call wait_for_all_tracers() in finally block

### Alternatives

- **Partial tracing**: Only workflow entry points (lose node visibility)
- **Custom metrics**: Build separate observability (more work)
- **No tracing**: Accept lack of visibility (poor debugging experience)

## Related Patterns

- [LangSmith Tracing Infrastructure](./langsmith-tracing-infrastructure.md) - Cost visibility foundation
- [LangSmith Batch Tracing](./langsmith-batch-tracing.md) - Batch API tracing
- [Conditional Development Tracing](../llm-interaction/conditional-development-tracing.md) - Dev/prod toggle
- [LangSmith Trace Identification](../langgraph/langsmith-trace-identification.md) - Semantic run_name

## Known Uses in Thala

- All 8 workflow entry points in `graph/api.py` files
- 40+ node functions across workflows
- `workflows/shared/tracing/tool_wrappers.py` - Tool wrapper implementations
- `workflows/shared/llm_utils/structured/executors/agent_runner.py` - Agent tool tracing
- All 15 test scripts in `testing/` directory with `wait_for_all_tracers()`

## References

- [LangSmith @traceable Decorator](https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code)
- [wait_for_all_tracers](https://api.python.langchain.com/en/latest/tracers/langchain_core.tracers.langchain.wait_for_all_tracers.html)
- [LangSmith Filtering](https://docs.smith.langchain.com/how_to_guides/tracing/filter_traces_in_application)
