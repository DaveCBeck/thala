---
name: content-series-generation
title: Content Series Generation with Distinctiveness Enforcement
date: 2026-01-26
category: langgraph
applicability:
  - "Generating multi-part content series from a single source document"
  - "Parallel content generation requiring cross-writer coordination"
  - "Workflows needing structured planning before parallel execution"
components: [workflow_graph, langgraph_node, llm_call, structured_output]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [parallel-execution, fan-out-fan-in, structured-output, content-generation, distinctiveness]
shared: true
gist_url: https://gist.github.com/DaveCBeck/769ad58a4252b346cfcada718679845c
article_path: .context/libs/thala-dev/content/2026-01-29-content-series-generation-langgraph.md
---

# Content Series Generation with Distinctiveness Enforcement

Transform a single literature review into a 4-part article series using parallel generation with cross-article distinctiveness enforcement.

## Problem

When generating a multi-part content series from a single source:
1. Articles tend to overlap thematically without explicit coordination
2. Parallel writers have no awareness of what others are covering
3. Sequential generation is slow for long-form content
4. Planning and writing require different model capabilities

## Solution

Use a LangGraph workflow with:
- **Structured planning** to assign distinct themes upfront
- **Parallel fetching** to gather content for each article
- **Sync barrier** to ensure all content is ready before writing
- **Distinctiveness enforcement** by passing "must_avoid" lists to parallel writers

## Workflow Architecture

```
START -> validate -> plan_content
      -> [3x fetch_content] -> sync_before_write
      -> [3x write_deep_dive] -> write_overview
      -> generate_images -> format_references -> END
```

### Phase Breakdown

| Phase | Nodes | Parallelism | Purpose |
|-------|-------|-------------|---------|
| Validate | `validate_input` | 1 | Extract citations, validate format |
| Plan | `plan_content` | 1 | Opus assigns themes with structured output |
| Fetch | `fetch_content` | 3 parallel via `Send()` | Get source content for each article |
| Sync | `sync_before_write` | 1 | Barrier ensures all fetches complete |
| Write | `write_deep_dive` | 3 parallel via `Send()` | Generate articles with distinctiveness |
| Finalize | `write_overview`, etc. | 1 | Overview, images, references |

## State Design

### Critical: Reducers for Parallel Writes

Any state key written by parallel nodes **must** have a reducer to avoid `INVALID_CONCURRENT_GRAPH_UPDATE`:

```python
from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

class EnrichedContent(TypedDict):
    """Content fetched from store for a deep-dive."""
    deep_dive_id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"]
    zotero_key: str
    content: str
    content_level: Literal["L0", "L2"]

class DeepDiveDraft(TypedDict):
    """A single deep-dive draft."""
    id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"]
    title: str
    content: str
    word_count: int
    citation_keys: list[str]

class EveningReadsState(TypedDict):
    """Main workflow state."""
    # Input
    input: dict

    # Planning phase (single writer, no reducer needed)
    deep_dive_assignments: list[dict]
    overview_scope: str

    # Fetching phase - PARALLEL, needs reducer
    enriched_content: Annotated[list[EnrichedContent], add]

    # Writing phase - PARALLEL, needs reducer
    deep_dive_drafts: Annotated[list[DeepDiveDraft], add]

    # Sequential phases (no reducer needed)
    overview_draft: Optional[dict]
    final_outputs: list[dict]

    # Error aggregation - PARALLEL, needs reducer
    errors: Annotated[list[dict], add]
```

## Structured Planning with Opus

The planning node uses structured output to ensure valid assignments:

```python
from pydantic import BaseModel, Field
from typing import Literal

class DeepDiveTopicPlan(BaseModel):
    """Plan for a single deep-dive article."""
    id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"]
    title: str = Field(description="Evocative, specific title (5-10 words)")
    theme: str = Field(description="2-3 sentence description of the theme")
    structural_approach: Literal["puzzle", "finding", "contrarian"]
    anchor_keys: list[str] = Field(description="2-3 citation keys that anchor this deep-dive")
    relevant_sections: list[str]
    distinctiveness_rationale: str

class PlanningOutput(BaseModel):
    """Structured output from planning."""
    deep_dives: list[DeepDiveTopicPlan] = Field(min_length=3, max_length=3)
    overview_scope: str
    series_coherence: str
```

### Structural Approaches

Each deep-dive gets a different narrative structure:

| Approach | Best For | Opening Strategy |
|----------|----------|------------------|
| `puzzle` | Mysteries, unexpected findings | Open with anomaly, unfold as investigation |
| `finding` | Data-driven topics | Lead with striking result, explore implications |
| `contrarian` | Overturning conventional wisdom | Steelman assumption, then complicate it |

## Parallel Fan-Out with Send()

### Fan-Out to Fetch

```python
from langgraph.types import Send

def route_to_fetch(state: EveningReadsState) -> list[Send] | str:
    """Fan out to parallel content fetching for each deep-dive."""
    assignments = state.get("deep_dive_assignments", [])
    citation_mappings = state.get("citation_mappings", {})

    if not assignments:
        return END

    sends = []
    for assignment in assignments:
        sends.append(
            Send(
                "fetch_content",
                {
                    "deep_dive_id": assignment["id"],
                    "anchor_keys": assignment["anchor_keys"],
                    "citation_mappings": citation_mappings,
                },
            )
        )
    return sends
```

### Sync Barrier

A pass-through node that waits for all parallel fetches to complete:

```python
async def sync_before_write_node(state: EveningReadsState) -> dict:
    """Synchronization barrier between fetch and write phases.

    All fetch nodes converge here via add_edge("fetch_content", "sync_before_write").
    This ensures enriched_content is fully aggregated before fanning out to writers.
    """
    enriched = state.get("enriched_content", [])
    logger.info(f"All fetches complete. Enriched content: {len(enriched)} items")
    return {}  # Pass-through, no state changes
```

### Fan-Out to Write with Distinctiveness

```python
def route_to_write(state: EveningReadsState) -> list[Send] | str:
    """Fan out to parallel deep-dive writing with distinctiveness enforcement."""
    assignments = state.get("deep_dive_assignments", [])
    enriched_content = state.get("enriched_content", [])
    lit_review = state["input"]["literature_review"]

    # Build must_avoid lists for distinctiveness
    # Each deep-dive should avoid the themes of the other two
    themes_by_id = {a["id"]: a["theme"] for a in assignments}

    sends = []
    for assignment in assignments:
        # Themes to avoid = all themes except this one
        must_avoid = [
            f"{other_id}: {theme}"
            for other_id, theme in themes_by_id.items()
            if other_id != assignment["id"]
        ]

        # Filter enriched content for this deep-dive
        dd_content = [
            ec for ec in enriched_content
            if ec["deep_dive_id"] == assignment["id"]
        ]

        sends.append(
            Send(
                "write_deep_dive",
                {
                    "deep_dive_id": assignment["id"],
                    "title": assignment["title"],
                    "theme": assignment["theme"],
                    "structural_approach": assignment["structural_approach"],
                    "must_avoid": must_avoid,  # Key for distinctiveness
                    "enriched_content": dd_content,
                    "literature_review": lit_review,
                },
            )
        )
    return sends
```

## Graph Construction

```python
from langgraph.graph import END, START, StateGraph

def create_evening_reads_graph() -> StateGraph:
    builder = StateGraph(EveningReadsState)

    # Add nodes
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("plan_content", plan_content_node)
    builder.add_node("fetch_content", fetch_content_node)
    builder.add_node("sync_before_write", sync_before_write_node)
    builder.add_node("write_deep_dive", write_deep_dive_node)
    builder.add_node("write_overview", write_overview_node)
    builder.add_node("generate_images", generate_images_node)
    builder.add_node("format_references", format_references_node)

    # Entry point
    builder.add_edge(START, "validate_input")

    # Conditional routing after validation
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        ["plan_content", END],
    )

    # Fan-out to fetch after planning
    builder.add_conditional_edges(
        "plan_content",
        route_to_fetch,
        ["fetch_content", END],
    )

    # All fetch nodes converge to sync barrier
    builder.add_edge("fetch_content", "sync_before_write")

    # Sync barrier fans out to writes
    builder.add_conditional_edges(
        "sync_before_write",
        route_to_write,
        ["write_deep_dive", END],
    )

    # All write nodes converge to overview
    builder.add_edge("write_deep_dive", "write_overview")

    # Linear flow to finish
    builder.add_edge("write_overview", "generate_images")
    builder.add_edge("generate_images", "format_references")
    builder.add_edge("format_references", END)

    return builder.compile()
```

## Distinctiveness Enforcement in Prompts

The `must_avoid` list is injected into the writing prompt:

```python
DEEP_DIVE_HEADER = """You are writing a deep-dive article...

## Your Deep-Dive Focus
Title: {title}
Theme: {theme}

## Must Avoid (covered in other deep-dives):
{must_avoid}

These themes are covered elsewhere in the series. Do NOT significantly
overlap with them. Brief mentions for context are fine, but the substance
of your piece must be distinct.
"""

# In the write node:
must_avoid_str = "\n".join(f"- {item}" for item in must_avoid)
system_prompt = prompt_template.format(
    title=title,
    theme=theme,
    must_avoid=must_avoid_str,
)
```

## Key Design Decisions

### Why Sync Barrier Instead of Direct Fan-In?

The sync barrier (`sync_before_write`) serves two purposes:
1. **Ensures complete aggregation**: All `enriched_content` must be collected before any writer starts
2. **Enables filtered distribution**: Each writer only receives content for their specific deep-dive

Without the barrier, a writer might start before all content is fetched.

### Why Structured Output for Planning?

- **Validation**: Pydantic enforces exactly 3 deep-dives, valid structural approaches
- **Type safety**: `anchor_keys` must be strings, IDs must be valid literals
- **Self-documentation**: Field descriptions guide the LLM's output

### Why Parallel Writes with Must-Avoid Instead of Sequential?

- **Speed**: 3 parallel writes complete in time of 1
- **Independence**: Each writer has complete context without waiting
- **Prompt-based coordination**: Cheaper than sequential chain-of-thought across articles

## Anti-Patterns to Avoid

### Missing Reducers

```python
# WRONG - Will cause INVALID_CONCURRENT_GRAPH_UPDATE
class State(TypedDict):
    deep_dive_drafts: list[dict]  # No reducer!

# CORRECT
class State(TypedDict):
    deep_dive_drafts: Annotated[list[dict], add]
```

### Fan-Out Without Convergence Point

```python
# WRONG - Results never collected
builder.add_conditional_edges("plan", route_to_fetch, ["fetch"])
builder.add_edge("fetch", END)  # Each fetch goes to END independently

# CORRECT - All fetches converge
builder.add_edge("fetch_content", "sync_before_write")
```

### Passing Full State to Send()

```python
# WRONG - Inefficient, passes entire state
Send("fetch_content", state)

# CORRECT - Pass only what the node needs
Send("fetch_content", {
    "deep_dive_id": assignment["id"],
    "anchor_keys": assignment["anchor_keys"],
})
```

## Testing Considerations

1. **Reducer aggregation**: Verify all parallel results are collected
2. **Distinctiveness**: Check that themes don't overlap significantly
3. **Structural variety**: Ensure different approaches produce different styles
4. **Error isolation**: One failed write shouldn't block others

## Related Patterns

- [Parallel Processing with Send()](./parallel-send.md) - General fan-out/fan-in
- [Structured Output Planning](./structured-output-planning.md) - Using Pydantic with LLMs
- [Sync Barriers in LangGraph](./sync-barriers.md) - Coordinating parallel phases

## Source

- Commit: `5dea359` - "feat(output): replace substack_review with evening_reads workflow"
- Files: `workflows/output/evening_reads/graph.py`, `state.py`, `nodes/`
