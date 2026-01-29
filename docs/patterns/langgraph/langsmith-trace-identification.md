---
name: langsmith-trace-identification
title: "LangSmith Trace Identification with run_name"
date: 2026-01-09
category: langgraph
applicability:
  - "Workflows with multiple parallel or sequential executions needing trace distinction"
  - "Debugging complex workflows with many invocations in LangSmith"
  - "Production systems requiring execution attribution and audit trails"
  - "Multi-loop supervision systems needing per-loop trace visibility"
components: [langgraph_config, run_id, run_name, workflow_api]
complexity: simple
verified_in_production: true
related_solutions: []
tags: [langgraph, langsmith, observability, tracing, debugging, run_name]
---

# LangSmith Trace Identification with run_name

## Intent

Add semantic naming to LangGraph workflow traces using `run_id` and `run_name` configuration, enabling instant visual identification in LangSmith's UI.

## Motivation

Without semantic naming, LangSmith traces appear as generic UUIDs, making it difficult to:
- Identify which workflow processed which input
- Debug failures without opening each trace
- Distinguish between parallel workflow executions
- Correlate user feedback to specific runs

This pattern adds meaningful names like `lit_review:Large Language Models` or `loop2_literature:AI Safety` that are immediately visible in the LangSmith trace list.

## Applicability

Use this pattern when:
- Running LangGraph workflows with LangSmith tracing enabled
- Multiple workflow types or invocations make traces hard to distinguish
- Debugging production issues requires quickly finding relevant traces
- Supervision loops need individual trace identification

Do NOT use this pattern when:
- LangSmith tracing is disabled (no benefit)
- Running single, isolated workflow tests (overkill)
- The workflow has no meaningful contextual identifier

## Structure

```
Workflow API Entry Point
    │
    ├─── Generate run_id: uuid.uuid4()
    │
    ├─── Extract context: topic[:30], theme[:30], etc.
    │
    ├─── Log run_id for debugging
    │
    └─── ainvoke() with config
              │
              ├─── run_id: UUID for LangSmith tracking
              │
              └─── run_name: "workflow_type:{context}"
```

## Implementation

### Step 1: Import UUID

```python
import uuid
import logging

logger = logging.getLogger(__name__)
```

### Step 2: Generate Run ID and Extract Context

```python
# Generate unique run ID
run_id = uuid.uuid4()

# Or convert from existing string ID in state
run_id = uuid.UUID(initial_state["langsmith_run_id"])

# Extract semantic context (truncated for UI)
context = topic[:30]  # Keep run_name readable

# Log for debugging/manual lookup
logger.info(f"LangSmith run ID: {run_id}")
```

### Step 3: Pass Config to ainvoke()

```python
result = await workflow_graph.ainvoke(
    initial_state,
    config={
        "run_id": run_id,
        "run_name": f"workflow_type:{context}",
    },
)
```

## Naming Conventions

| Workflow | Pattern | Example |
|----------|---------|---------|
| Academic Literature Review | `lit_review:{topic}` | `lit_review:Large Language Models in` |
| Book Finding | `books:{theme}` | `books:Scientific Discovery Methods` |
| Document Processing | `doc:{title}` | `doc:Research Paper Analysis` |
| Supervised Lit Review | `supervised_lit:{topic}` | `supervised_lit:Machine Learning` |
| Loop 1 (Theoretical Depth) | `loop1_theory:{topic}` | `loop1_theory:AI Safety` |
| Loop 2 (Literature) | `loop2_literature:{topic}` | `loop2_literature:Neural Networks` |
| Loop 3 (Structure) | `loop3_structure:{topic}` | `loop3_structure:Deep Learning` |
| Loop 4 (Editing) | `loop4_editing:{topic}` | `loop4_editing:Transformers` |
| Loop 5 (Fact Check) | `loop5_factcheck:{topic}` | `loop5_factcheck:LLM Applications` |

**Naming principles:**
- Lowercase with underscores: `loop1_theory`, not `Loop1_Theory`
- Workflow type as prefix: `lit_review:`, `books:`, `doc:`
- Colon separator: `workflow_type:{identifier}`
- Always truncate: `identifier[:30]` for readability

## Complete Example

```python
"""
Academic literature review with LangSmith trace identification.
"""
import logging
import uuid
from typing import Literal

from workflows.academic_lit_review.graph.construction import academic_lit_review_graph
from workflows.academic_lit_review.state import build_initial_state

logger = logging.getLogger(__name__)


async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: Literal["test", "quick", "standard", "comprehensive", "high_quality"] = "standard",
    language: str = "en",
) -> dict:
    """Run academic literature review with named LangSmith traces."""

    # Build initial state
    initial_state = build_initial_state(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        language=language,
    )

    # Extract or generate run ID
    run_id = uuid.UUID(initial_state["langsmith_run_id"])
    logger.info(f"Starting lit review: {topic}")
    logger.info(f"LangSmith run ID: {run_id}")

    try:
        # Invoke with named trace
        result = await academic_lit_review_graph.ainvoke(
            initial_state,
            config={
                "run_id": run_id,
                "run_name": f"lit_review:{topic[:30]}",
            },
        )

        final_review = result.get("final_review", "")
        errors = result.get("errors", [])

        # Determine standardized status
        if final_review and not errors:
            status = "success"
        elif final_review and errors:
            status = "partial"
        else:
            status = "failed"

        return {
            "final_review": final_review,
            "status": status,
            "langsmith_run_id": str(run_id),
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"Literature review failed: {e}")
        return {
            "final_review": f"Literature review generation failed: {e}",
            "status": "failed",
            "langsmith_run_id": str(run_id),
            "errors": [{"phase": "unknown", "error": str(e)}],
        }
```

### Per-Loop Naming for Supervision

```python
"""
Supervision loop with individual trace names.
"""
import uuid

async def run_loop2_node(state: OrchestrationState) -> dict:
    """Run Loop 2: Literature base expansion."""

    topic = state["input"].get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop2_standalone(
        review=state["current_review"],
        paper_corpus=state["paper_corpus"],
        paper_summaries=state["paper_summaries"],
        input_data=state["input"],
        quality_settings=state["quality_settings"],
        max_iterations=state["loop_progress"]["max_iterations_per_loop"],
        config={
            "run_id": loop_run_id,
            "run_name": f"loop2_literature:{topic}",
        },
    )

    return {
        "current_review": result.get("current_review", state["current_review"]),
        "paper_corpus": result.get("paper_corpus", state["paper_corpus"]),
        "loop2_result": result,
    }
```

## Consequences

### Benefits

- **Instant identification**: Traces visible by name in LangSmith UI
- **Reduced debugging time**: Find relevant traces without opening each one
- **Audit trail**: Link user feedback to specific workflow runs
- **Loop visibility**: Supervision loops show as separate named traces
- **Minimal overhead**: Just uuid generation and f-string formatting

### Trade-offs

- **Requires semantic context**: Need meaningful identifier (topic, theme, title)
- **Truncation needed**: Long identifiers must be cut for readability
- **UUID management**: Must handle string/UUID conversion consistently

### Alternatives

- **Tags only**: Use LangSmith tags instead of run_name (less visible in UI)
- **Metadata only**: Store context in metadata (requires clicking into trace)
- **No naming**: Accept generic UUIDs (sufficient for simple use cases)

## Related Patterns

- [Conditional Development Tracing](../llm-interaction/conditional-development-tracing.md) - Environment-based tracing setup
- [Multi-Loop Supervision System](./multi-loop-supervision-system.md) - Where per-loop naming applies

## Known Uses in Thala

- `workflows/academic_lit_review/graph/api.py`: `lit_review:{topic[:30]}`
- `workflows/book_finding/graph/api.py`: `books:{theme[:30]}`
- `workflows/document_processing/graph.py`: `doc:{desc[:30]}`
- `workflows/supervised_lit_review/api.py`: `supervised_lit:{topic}`
- `workflows/supervised_lit_review/supervision/orchestration/graph.py`: Per-loop naming (loop1-5)
- `workflows/web_research/graph/api.py`: `deep_research:{query[:30]}`

## References

- [LangSmith Tracing Concepts](https://docs.smith.langchain.com/concepts/tracing)
- [LangGraph Configuration](https://langchain-ai.github.io/langgraph/concepts/configuration/)
