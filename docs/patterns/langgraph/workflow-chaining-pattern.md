---
name: workflow-chaining-pattern
title: "Workflow Chaining: Sequential Workflow Composition"
date: 2026-01-16
category: langgraph
applicability:
  - "Multi-phase processing where outputs of one workflow feed into another"
  - "Document enhancement requiring both content enrichment and structural editing"
  - "Pipelines with optional phases that can be toggled independently"
  - "Error-resilient processing where phase failures don't block entire pipeline"
components: [orchestrator, state_handoff, phase_toggle, quality_propagation]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [workflow-composition, chaining, sequential, orchestration, supervision, editing]
---

# Workflow Chaining: Sequential Workflow Composition

## Intent

Compose multiple independent workflows into a sequential pipeline where each phase's output becomes the next phase's input, with optional phase toggling and graceful error handling.

## Motivation

Complex document processing often requires multiple distinct transformation phases:
- **Supervision**: Content enrichment (adding evidence, citations, depth)
- **Editing**: Structural improvement (organization, flow, polish)

These concerns are best implemented as separate workflows that can be:
- Run independently for debugging
- Toggled on/off for different use cases
- Chained together for full processing

This pattern provides the orchestration mechanics for composing workflows sequentially.

## Applicability

Use this pattern when:
- Multiple workflows need to run in sequence
- Each workflow can be independently tested and debugged
- Phases should be toggleable (skip supervision, skip editing)
- Quality settings should propagate across all phases
- Phase failures shouldn't block the entire pipeline

Do NOT use this pattern when:
- Workflows must run in parallel (use multi-source orchestration instead)
- Deep state sharing is required between workflows
- Tight coupling between phases is acceptable

## Structure

```
                    enhance_report()
                          │
     ┌────────────────────┼────────────────────┐
     │                    │                    │
     ▼                    ▼                    ▼
┌─────────┐         ┌─────────┐         ┌─────────┐
│ loops=  │         │ loops=  │         │ loops=  │
│ "none"  │         │ "one"   │         │ "all"   │
└────┬────┘         └────┬────┘         └────┬────┘
     │                   │                   │
     │              ┌────┴────┐         ┌────┴────┐
     │              │ Loop 1  │         │ Loop 1  │
     │              │ Theory  │         │ Theory  │
     │              └────┬────┘         └────┬────┘
     │                   │              ┌────┴────┐
     │                   │              │ Loop 2  │
     │                   │              │ Expand  │
     │                   │              └────┬────┘
     │                   │                   │
     ├───────────────────┴───────────────────┤
     │                                       │
     ▼ (if run_editing=True)                 ▼
┌─────────────────────────────────────────────────┐
│              Editing Workflow                    │
│  ┌──────────┬───────────┬──────────┬─────────┐  │
│  │Structure │Enhancement│Verification│ Polish │  │
│  └──────────┴───────────┴──────────┴─────────┘  │
└─────────────────────────────────────────────────┘
                          │
                          ▼
                   Final Document
```

## Implementation

### Step 1: Define Orchestrator API

```python
# workflows/enhance/__init__.py

from typing import Literal, Optional, Any

async def enhance_report(
    report: str,
    topic: str,
    research_questions: list[str],
    quality: Literal["quick", "standard", "comprehensive", "high_quality"] = "standard",
    loops: Literal["none", "one", "two", "all"] = "all",
    run_editing: bool = True,
    paper_corpus: Optional[dict[str, Any]] = None,
    paper_summaries: Optional[dict[str, Any]] = None,
    zotero_keys: Optional[dict[str, str]] = None,
    config: Optional[dict] = None,
) -> dict[str, Any]:
    """Orchestrate supervision and editing workflows in sequence.

    Args:
        report: Input markdown document
        topic: Document topic for context
        research_questions: Research questions guiding content
        quality: Quality tier (affects both phases)
        loops: Which supervision loops to run ("none", "one", "two", "all")
        run_editing: Whether to run editing phase after supervision
        paper_corpus: Optional pre-existing paper corpus
        paper_summaries: Optional pre-existing summaries
        zotero_keys: Optional pre-existing citation keys
        config: Optional LangSmith config

    Returns:
        Dict with final_report, status, and phase-specific results
    """
```

### Step 2: Implement Sequential Execution

```python
async def enhance_report(...) -> dict[str, Any]:
    errors = []
    current_report = report
    supervision_result = None
    editing_result = None

    # Phase 1: Supervision (conditional)
    if loops != "none":
        logger.info(f"Phase 1: Running supervision loops={loops}")
        try:
            supervision_result = await supervision_enhance(
                report=current_report,
                topic=topic,
                research_questions=research_questions,
                quality=quality,
                loops=loops,
                paper_corpus=paper_corpus or {},
                paper_summaries=paper_summaries or {},
                zotero_keys=zotero_keys or {},
                config=config,
            )

            # Extract outputs for next phase
            current_report = supervision_result["final_report"]
            paper_corpus = supervision_result["paper_corpus"]
            paper_summaries = supervision_result["paper_summaries"]
            zotero_keys = supervision_result["zotero_keys"]

            logger.info(
                f"Supervision complete: {len(current_report)} chars, "
                f"{len(paper_corpus)} papers"
            )

        except Exception as e:
            logger.error(f"Supervision failed: {e}")
            errors.append({"phase": "supervision", "error": str(e)})
            # Continue with original report

    # Phase 2: Editing (conditional)
    if run_editing:
        logger.info("Phase 2: Running editing workflow")
        try:
            editing_result = await editing(
                document=current_report,
                topic=topic,
                quality=quality,
                config=config,
            )

            if editing_result.get("status") != "failed":
                current_report = editing_result["final_report"]

        except Exception as e:
            logger.error(f"Editing failed: {e}")
            errors.append({"phase": "editing", "error": str(e)})

    # Determine overall status
    status = "failed" if not current_report else (
        "partial" if errors else "success"
    )

    return {
        "final_report": current_report,
        "status": status,
        "supervision_result": supervision_result,
        "editing_result": editing_result,
        "paper_corpus": paper_corpus,
        "paper_summaries": paper_summaries,
        "zotero_keys": zotero_keys,
        "errors": errors,
    }
```

### Step 3: Quality Propagation

```python
# Both phases receive the same quality tier
# Each phase interprets it according to its own presets

# Supervision quality (workflows/enhance/supervision/README.md)
SUPERVISION_QUALITY = {
    "quick": {"max_stages": 2, "max_papers": 50},
    "standard": {"max_stages": 3, "max_papers": 100},
    "comprehensive": {"max_stages": 4, "max_papers": 200},
    "high_quality": {"max_stages": 5, "max_papers": 300},
}

# Editing quality (workflows/enhance/editing/quality_presets.py)
EDITING_QUALITY = {
    "quick": {"max_structure_iterations": 2, "max_polish_edits": 5},
    "standard": {"max_structure_iterations": 3, "max_polish_edits": 10},
    "comprehensive": {"max_structure_iterations": 4, "max_polish_edits": 15},
    "high_quality": {"max_structure_iterations": 5, "max_polish_edits": 20},
}
```

### Step 4: State Handoff

```python
# Supervision outputs
supervision_result = {
    "final_report": str,           # → becomes editing input
    "paper_corpus": dict,          # → preserved for re-runs
    "paper_summaries": dict,       # → preserved for re-runs
    "zotero_keys": dict,           # → preserved for re-runs
    "loops_run": list[str],        # → audit trail
    "loop1_result": dict | None,   # → audit trail
    "loop2_result": dict | None,   # → audit trail
}

# Editing receives minimal handoff
editing_input = {
    "document": supervision_result["final_report"],  # The key handoff
    "topic": topic,                                   # Shared context
    "quality": quality,                               # Shared config
}

# Editing doesn't need paper_corpus - it auto-detects citations
```

## Complete Example

```python
from workflows.enhance import enhance_report

# Full pipeline: supervision + editing
result = await enhance_report(
    report=markdown_text,
    topic="Machine Learning in Healthcare",
    research_questions=[
        "How effective is ML for diagnosis?",
        "What are the main limitations?",
    ],
    quality="standard",
    loops="all",
    run_editing=True,
)

print(f"Status: {result['status']}")
print(f"Final length: {len(result['final_report'])} chars")
print(f"Papers discovered: {len(result['paper_corpus'])}")

# Access phase-specific results
if result["supervision_result"]:
    print(f"Loops run: {result['supervision_result']['loops_run']}")

if result["editing_result"]:
    print(f"Editing status: {result['editing_result']['status']}")
```

### Usage Variants

```python
# Supervision only
result = await enhance_report(
    report=text, topic="...", research_questions=[...],
    loops="all", run_editing=False,
)

# Editing only
result = await enhance_report(
    report=text, topic="...", research_questions=[...],
    loops="none", run_editing=True,
)

# Loop 1 only + editing
result = await enhance_report(
    report=text, topic="...", research_questions=[...],
    loops="one", run_editing=True,
)

# Reuse existing paper corpus
result = await enhance_report(
    report=text, topic="...", research_questions=[...],
    paper_corpus=previous_result["paper_corpus"],
    paper_summaries=previous_result["paper_summaries"],
    zotero_keys=previous_result["zotero_keys"],
)
```

## Consequences

### Benefits

- **Modularity**: Each workflow is independently testable
- **Flexibility**: Phases can be toggled for different use cases
- **Error resilience**: Phase failures don't block the entire pipeline
- **Auditability**: Phase-specific results preserved in output
- **Corpus accumulation**: Papers discovered persist across runs

### Trade-offs

- **Sequential latency**: Phases run one after another (not parallelized)
- **Loose coupling**: Limited state sharing between phases
- **Quality interpretation**: Each phase interprets quality tier independently
- **Error recovery**: Manual intervention needed for partial failures

### Alternatives

- **Monolithic workflow**: Single graph with all phases (tighter coupling)
- **Parallel composition**: Run phases in parallel (for independent outputs)
- **Event-driven**: Message queue between phases (for distributed systems)

## Related Patterns

- [Workflow Modularization Pattern](./workflow-modularization-pattern.md) - How to structure composable workflows
- [Multi-Source Research Orchestration](./multi-source-research-orchestration.md) - Parallel workflow composition
- [Multi-Loop Supervision System](./multi-loop-supervision-system.md) - Sequential loop chaining
- [Multi-Phase Document Editing](./multi-phase-document-editing.md) - Editing workflow details

## Known Uses in Thala

- `workflows/enhance/__init__.py` - Main orchestrator
- `workflows/enhance/supervision/` - Supervision workflow (Loop 1 + Loop 2)
- `workflows/enhance/editing/` - Editing workflow (4 phases)

## References

- [LangGraph Workflow Composition](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [Python async/await](https://docs.python.org/3/library/asyncio.html)
