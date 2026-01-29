---
name: fact-check-workflow-extraction
title: "Fact-Check Workflow Extraction: Standalone Verification Phase"
date: 2026-01-28
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/e55f9b044f5f02b502be44abc61aabbd
article_path: .context/libs/thala-dev/content/2026-01-28-fact-check-workflow-extraction-langgraph.md
applicability:
  - "Verification functionality embedded in larger editing workflows"
  - "Multi-phase document pipelines needing independent phase toggles"
  - "Workflows requiring parallel section-level workers with result aggregation"
  - "Quality-tiered verification with configurable confidence thresholds"
components: [standalone_workflow, parallel_section_workers, accumulator_reducers, citation_gating, quality_presets]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [workflow-extraction, fact-checking, verification, parallel-workers, send-pattern, reducers, three-phase-pipeline]
---

# Fact-Check Workflow Extraction: Standalone Verification Phase

## Intent

Extract specialized verification functionality from a monolithic editing workflow into a standalone workflow with parallel section-level workers, enabling independent execution, quality tier configuration, and clean integration as a composable pipeline phase.

## Motivation

When verification logic grows complex—multiple phases (screening, fact-checking, reference-checking), parallel workers, and sophisticated result aggregation—embedding it in a larger editing workflow creates problems:

**The Problem:**
```
editing/
├── nodes/
│   ├── structure.py          (500 lines)
│   ├── enhance.py            (400 lines)
│   ├── verify.py             (800+ lines)  ← Growing complexity
│   │   ├── screen_sections()
│   │   ├── fact_check_worker()
│   │   ├── reference_check_worker()
│   │   ├── apply_edits()
│   │   └── aggregate_results()
│   └── polish.py             (200 lines)
└── graph/
    └── construction.py        (Complex routing for verify phase embedded)

Issues:
- Cannot run verification independently
- Cannot skip verification without code changes
- Cannot test verification in isolation
- Quality settings mixed with editing settings
- Routing logic for parallel workers clutters main graph
```

**The Solution:**
```
enhance/
├── editing/                   # Clean editing workflow
│   ├── nodes/
│   └── graph/
├── fact_check/                # Standalone verification workflow
│   ├── state.py               # Independent state schema
│   ├── schemas.py             # Pydantic models for LLM outputs
│   ├── quality_presets.py     # Verification-specific presets
│   ├── prompts.py             # LLM prompts
│   ├── nodes/                 # 7 specialized nodes
│   └── graph/                 # Self-contained routing
└── __init__.py                # Three-phase orchestration

Benefits:
- Run fact_check() independently for any document
- Toggle with run_fact_check=True/False
- Test verification in isolation
- Quality presets control verification depth
- Clean parent orchestration: supervision → editing → fact_check
```

## Applicability

Use this pattern when:
- Verification functionality has grown to 500+ lines
- Multiple parallel workers needed for section-level processing
- Need to run verification independently of editing
- Quality tiers should control verification depth separately
- Phase should be toggleable without modifying core workflow

Do NOT use this pattern when:
- Verification is simple (single pass, no parallelism)
- Tight coupling with editing is intentional
- Overhead of separate workflow not justified (< 200 lines)
- No need for independent quality configuration

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  enhance_report() - Three-Phase Orchestration                      │
│                                                                    │
│  Phase 1: supervision_enhance()  ─┐                                │
│           (optional, loops param)  │ Sequential                    │
│                                   ↓ direct calls                   │
│  Phase 2: editing()            ────┤                               │
│           (optional, run_editing)  │                               │
│                                   ↓                                │
│  Phase 3: fact_check()         ────┘                               │
│           (optional, run_fact_check)                               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  fact_check() - Standalone Verification Workflow                   │
│                                                                    │
│  ┌──────────────┐                                                  │
│  │ parse_document│ ← Reuse pre-parsed model if available           │
│  └──────┬───────┘                                                  │
│         ▼                                                          │
│  ┌──────────────┐                                                  │
│  │detect_citations│                                                │
│  └──────┬───────┘                                                  │
│         │                                                          │
│         ▼ [has_citations?]                                         │
│    ┌────┴────┐                                                     │
│    │ No      │ Yes                                                 │
│    ▼         ▼                                                     │
│  finalize   screen_sections                                        │
│  (skip)     ┌──────┬──────┐                                        │
│             ▼      ▼      ▼                                        │
│         ┌──────┐┌──────┐┌──────┐   ← Parallel via Send()          │
│         │fact_ ││fact_ ││fact_ │                                   │
│         │check ││check ││check │                                   │
│         │sec1  ││sec2  ││sec3  │                                   │
│         └──┬───┘└──┬───┘└──┬───┘                                   │
│            └───────┼───────┘                                       │
│                    ▼                                               │
│         ┌──────────────────┐                                       │
│         │assemble_fact_    │  ← Reducer aggregates results         │
│         │checks            │                                       │
│         └────────┬─────────┘                                       │
│                  ▼                                                 │
│         ┌──────────────────┐                                       │
│         │pre_validate_     │  ← Cache citation existence           │
│         │citations         │                                       │
│         └────────┬─────────┘                                       │
│                  ▼                                                 │
│         ┌──────┐┌──────┐┌──────┐   ← Parallel via Send()          │
│         │ref_  ││ref_  ││ref_  │                                   │
│         │check ││check ││check │                                   │
│         │sec1  ││sec2  ││sec3  │                                   │
│         └──┬───┘└──┬───┘└──┬───┘                                   │
│            └───────┼───────┘                                       │
│                    ▼                                               │
│         ┌──────────────────┐                                       │
│         │apply_verified_   │                                       │
│         │edits             │                                       │
│         └────────┬─────────┘                                       │
│                  ▼                                                 │
│         ┌──────────────────┐                                       │
│         │finalize          │                                       │
│         └──────────────────┘                                       │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Create Standalone Workflow Structure

```python
# workflows/enhance/fact_check/__init__.py

"""Standalone fact-checking workflow for document verification."""

from .graph.api import fact_check
from .state import FactCheckState, build_initial_state
from .quality_presets import QUALITY_PRESETS, QualitySettings

__all__ = [
    "fact_check",
    "FactCheckState",
    "build_initial_state",
    "QUALITY_PRESETS",
    "QualitySettings",
]
```

### Step 2: Define State with Accumulator Reducers

```python
# workflows/enhance/fact_check/state.py

from operator import add
from typing import Annotated, Any, Literal, TypedDict

class FactCheckState(TypedDict, total=False):
    """State for fact-check workflow with parallel worker support."""

    # === Input ===
    input: dict  # document, document_model, topic
    quality_settings: dict[str, Any]

    # === Parse Phase (sequential) ===
    document_model: dict
    parse_complete: bool

    # === Citation Detection (sequential) ===
    has_citations: bool
    citation_keys: list[str]

    # === Screening (sequential) ===
    screened_sections: list[str]  # Section IDs to check
    screening_skipped: list[str]  # Section IDs skipped

    # === Fact-Check (parallel workers) ===
    # Annotated[list, add] accumulates results from parallel workers
    fact_check_results: Annotated[list[dict], add]

    # === Reference-Check (parallel workers) ===
    citation_cache: dict[str, dict]  # Pre-validated cache
    reference_check_results: Annotated[list[dict], add]

    # === Apply Edits (sequential) ===
    pending_edits: Annotated[list[dict], add]  # From both phases
    applied_edits: list[dict]
    skipped_edits: list[dict]
    unresolved_items: Annotated[list[dict], add]

    # === Output ===
    final_document: str
    status: Literal["success", "partial", "failed", "skipped"]
    errors: Annotated[list[dict], add]
```

**Key Pattern:** `Annotated[list[dict], add]` enables parallel workers to return `{"fact_check_results": [result]}` and have LangGraph automatically concatenate all results.

### Step 3: Implement Parallel Worker Dispatch with Send()

```python
# workflows/enhance/fact_check/nodes/fact_check.py

from langgraph.types import Send

def route_to_fact_check_sections(state: dict) -> list[Send] | str:
    """Route to parallel fact-check workers or skip to next phase."""
    screened_sections = state.get("screened_sections", [])
    document_model = deserialize_document_model(state.get("document_model", {}))
    quality_settings = state.get("quality_settings", {})

    sections_to_check = [
        s for s in document_model.get_leaf_sections()
        if s.section_id in screened_sections
    ]

    if not sections_to_check:
        return "reference_check_router"  # Skip to next phase

    # Dispatch parallel workers
    sends = []
    for section in sections_to_check:
        sends.append(Send("fact_check_section", {
            "section_id": section.section_id,
            "section_content": document_model.get_section_content(section.section_id),
            "section_heading": section.heading,
            "use_perplexity": quality_settings.get("perplexity_enabled", True),
            "confidence_threshold": quality_settings.get(
                "verify_confidence_threshold", 0.75
            ),
            "max_tool_calls": quality_settings.get("fact_check_max_tool_calls", 15),
        }))

    return sends


@traceable(run_type="chain", name="FactCheckSectionWorker")
async def fact_check_section_worker(state: dict) -> dict[str, Any]:
    """Fact-check a single section. Returns results for accumulation."""
    section_id = state["section_id"]
    content = state["section_content"]
    confidence_threshold = state.get("confidence_threshold", 0.75)

    # ... fact-check logic with LLM and tools ...

    # Return list for add reducer to accumulate
    return {
        "fact_check_results": [result.model_dump()],
        "pending_edits": [e.model_dump() for e in valid_edits],
    }
```

### Step 4: Wire Graph with Conditional Routing

```python
# workflows/enhance/fact_check/graph/construction.py

from langgraph.graph import StateGraph, START, END

def build_fact_check_graph() -> StateGraph:
    """Build the fact-check workflow graph."""
    builder = StateGraph(FactCheckState)

    # Add nodes
    builder.add_node("parse_document", parse_document_node)
    builder.add_node("detect_citations", detect_citations_node)
    builder.add_node("screen_fact_check", screen_sections_node)
    builder.add_node("fact_check_section", fact_check_section_worker)
    builder.add_node("assemble_fact_checks", assemble_fact_checks_node)
    builder.add_node("pre_validate_citations", pre_validate_citations_node)
    builder.add_node("reference_check_section", reference_check_section_worker)
    builder.add_node("assemble_reference_checks", assemble_reference_checks_node)
    builder.add_node("apply_verified_edits", apply_verified_edits_node)
    builder.add_node("finalize", finalize_node)

    # Sequential start
    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "detect_citations")

    # Citation gating: skip entire workflow if no citations
    builder.add_conditional_edges(
        "detect_citations",
        route_citations_or_finalize,
        {"screen_fact_check": "screen_fact_check", "finalize": "finalize"},
    )

    # Parallel fact-check dispatch
    builder.add_conditional_edges(
        "screen_fact_check",
        route_to_fact_check_sections,
        ["fact_check_section", "reference_check_router"],
    )

    # Workers converge to assembly
    builder.add_edge("fact_check_section", "assemble_fact_checks")
    builder.add_edge("assemble_fact_checks", "pre_validate_citations")

    # Parallel reference-check dispatch
    builder.add_edge("pre_validate_citations", "reference_check_router")
    builder.add_conditional_edges(
        "reference_check_router",
        route_to_reference_check_sections,
        ["reference_check_section", "apply_verified_edits"],
    )

    builder.add_edge("reference_check_section", "assemble_reference_checks")
    builder.add_edge("assemble_reference_checks", "apply_verified_edits")
    builder.add_edge("apply_verified_edits", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
```

### Step 5: Integrate as Pipeline Phase

```python
# workflows/enhance/__init__.py

async def enhance_report(
    report: str,
    topic: str,
    quality: str = "standard",
    loops: str = "both",
    run_editing: bool = True,
    run_fact_check: bool = True,  # Toggle fact-check phase
) -> dict[str, Any]:
    """Three-phase enhancement: supervision → editing → fact_check."""

    current_report = report

    # Phase 1: Supervision (optional)
    if loops != "none":
        supervision_result = await supervision_enhance(
            report=current_report, topic=topic, quality=quality, loops=loops
        )
        current_report = supervision_result["final_report"]

    # Phase 2: Editing (optional)
    if run_editing:
        editing_result = await editing(
            document=current_report, topic=topic, quality=quality
        )
        current_report = editing_result["final_report"]

    # Phase 3: Fact-Check (optional, standalone workflow)
    fact_check_result = None
    if run_fact_check:
        fact_check_result = await fact_check(
            document=current_report,
            topic=topic,
            quality=quality,
            # Optional: pass pre-computed state from editing
            # document_model=editing_result.get("document_model"),
            # has_citations=editing_result.get("has_citations"),
        )
        current_report = fact_check_result["final_report"]

    return {
        "final_report": current_report,
        "fact_check_result": fact_check_result,
        "status": determine_overall_status(...),
    }
```

## Complete Example

```python
# Standalone usage
from workflows.enhance.fact_check import fact_check

result = await fact_check(
    document="""
    # Climate Change Analysis

    Recent studies show global temperatures have risen by 1.5°C [@Smith2023].
    This finding contradicts earlier predictions [@Jones2020].
    """,
    topic="climate change impacts",
    quality="standard",  # Controls verification depth
)

print(f"Status: {result['status']}")
print(f"Claims checked: {len(result['fact_check_results'])}")
print(f"Citations validated: {len(result['reference_check_results'])}")
print(f"Edits applied: {len(result['applied_edits'])}")
print(f"Unresolved issues: {len(result['unresolved_items'])}")

# As part of pipeline
from workflows.enhance import enhance_report

result = await enhance_report(
    report=raw_document,
    topic="climate change",
    quality="comprehensive",
    loops="both",           # Run supervision loops
    run_editing=True,       # Run editing phase
    run_fact_check=True,    # Run fact-check phase
)
```

## Consequences

### Benefits

- **Independent execution**: Run fact-check on any document without editing
- **Phase toggling**: `run_fact_check=False` cleanly skips the phase
- **Isolated testing**: Test verification logic without editing dependencies
- **Quality tier control**: Verification-specific presets (confidence thresholds, tool calls)
- **Parallel efficiency**: Section-level workers maximize throughput
- **Clean aggregation**: Add reducers handle parallel result collection automatically

### Trade-offs

- **Additional coordination**: State must be threaded between phases
- **Potential re-parsing**: If editing doesn't expose document_model, fact-check re-parses
- **Workflow overhead**: Separate graph construction for simple use cases
- **Configuration duplication**: Quality presets in multiple workflow packages

### Alternatives

- **Embedded verification**: Keep verification nodes in editing graph (simpler, less flexible)
- **Subgraph composition**: Use LangGraph subgraph instead of standalone workflow (shared state schema required)
- **Event-driven**: Publish/subscribe between phases (complex, overkill for sequential)

## Related Patterns

- [Workflow Modularization Pattern](./workflow-modularization-pattern.md) - General guidance on extracting workflows
- [Workflow Chaining Pattern](./workflow-chaining-pattern.md) - Sequential composition of independent workflows
- [Multi-Phase Document Editing](./multi-phase-document-editing.md) - Four-phase editing with embedded verification
- [Section Rewriting and Citation Validation](./section-rewriting-citation-validation.md) - Zotero-based citation verification

## Known Uses in Thala

- `workflows/enhance/fact_check/` - Full standalone fact-checking workflow
- `workflows/enhance/fact_check/graph/api.py` - Public `fact_check()` entry point
- `workflows/enhance/fact_check/nodes/fact_check.py` - Parallel worker dispatch with Send()
- `workflows/enhance/fact_check/state.py` - Accumulator reducers for parallel results
- `workflows/enhance/__init__.py` - Three-phase orchestration (supervision → editing → fact_check)

## References

- [LangGraph Send Pattern](https://langchain-ai.github.io/langgraph/how-tos/send/)
- [LangGraph State Reducers](https://langchain-ai.github.io/langgraph/concepts/#reducers)
- [Parallel Node Execution](https://langchain-ai.github.io/langgraph/how-tos/parallel/)
