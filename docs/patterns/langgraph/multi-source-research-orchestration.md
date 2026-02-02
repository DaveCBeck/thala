---
name: multi-source-research-orchestration
title: Multi-Source Research Orchestration Pattern
date: 2026-01-02
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/2311e637dde740636b596fa4b48ba0b0
article_path: .context/libs/thala-dev/content/2026-01-02-multi-workflow-orchestration-langgraph.md
applicability:
  - "Research requiring multiple source types (web, academic, books)"
  - "Parallel workflow execution with independent failure handling"
  - "Unified quality tier mapping across heterogeneous workflows"
  - "Long-running workflows needing checkpoint/resume support"
components: [wrapped, deep_research, academic_lit_review, book_finding]
complexity: high
verified_in_production: false
deprecated: true
deprecated_date: 2026-01-29
deprecated_reason: "workflows/wrapped/ directory deleted in commit ed6c2c5; architecture reorganized"
superseded_by: "./synthesis-workflow-orchestration.md"
tags: [orchestration, parallel-execution, asyncio, checkpointing, quality-tiers, multi-workflow]
---

> **DEPRECATED**: This documentation describes code that has been removed.
>
> **Reason:** The `workflows/wrapped/` directory was deleted during reorganization (commit ed6c2c5). Multi-source orchestration is now handled differently.
> **Date:** 2026-01-29
> **See instead:** [Synthesis Workflow Orchestration](./synthesis-workflow-orchestration.md)

# Multi-Source Research Orchestration Pattern

## Intent

Orchestrate multiple specialized research workflows (web, academic, books) running in parallel, with unified quality settings, independent failure handling, and checkpoint support for resumption after interruption.

## Problem

Complex research topics benefit from multiple source types:
- **Web research**: Current information, recent developments, diverse perspectives
- **Academic literature**: Peer-reviewed depth, methodology, theoretical foundations
- **Books**: Cross-domain connections, transformative ideas, deeper exploration

Running these separately is inefficient and requires manual coordination. Running them together requires handling:
- Different quality/depth parameters per workflow
- Independent failures (one failing shouldn't stop others)
- Long execution times (hours) needing checkpoints
- Combining outputs into coherent summary

## Solution

Create a wrapper workflow that:
1. Maps unified quality tiers to each sub-workflow's parameters
2. Runs web + academic in parallel via `asyncio.gather`
3. Generates thematic book queries from research findings
4. Calls book_finding with quality-appropriate settings
5. Saves individual + combined results to storage
6. Supports file-based checkpointing for resumption

## Structure

```
workflows/wrapped/
├── __init__.py           # Public API exports
├── state.py              # State with quality mapping
├── checkpointing.py      # File-based checkpoint utilities
├── graph/
│   ├── __init__.py
│   ├── construction.py   # LangGraph builder
│   └── api.py            # wrapped_research() entry point
└── nodes/
    ├── __init__.py
    ├── run_parallel_research.py  # asyncio.gather web + academic
    ├── generate_book_query.py    # LLM generates book theme
    ├── run_book_finding.py       # Book finding workflow
    ├── generate_final_summary.py # LLM synthesizes all sources
    └── save_to_top_of_mind.py    # Save 4 records to storage
```

## Implementation

### Quality Tier Mapping

Map unified quality levels to each workflow's parameters:

```python
# workflows/wrapped/state.py

from typing import Literal
from typing_extensions import TypedDict

QualityTier = Literal["quick", "standard", "comprehensive"]

# Unified quality maps to each workflow's settings
QUALITY_MAPPING: dict[str, dict[str, str]] = {
    "quick": {
        "web_depth": "quick",
        "academic_quality": "quick",
        "book_quality": "quick",
    },
    "standard": {
        "web_depth": "standard",
        "academic_quality": "standard",
        "book_quality": "standard",
    },
    "comprehensive": {
        "web_depth": "comprehensive",
        "academic_quality": "high_quality",  # Academic uses different naming
        "book_quality": "comprehensive",
    },
}


class WrappedResearchInput(TypedDict):
    """Input parameters for wrapped research workflow."""
    query: str
    quality: QualityTier
    research_questions: Optional[list[str]]  # For academic
    date_range: Optional[tuple[int, int]]    # For academic


class WorkflowResult(TypedDict):
    """Result from a single sub-workflow."""
    workflow_type: str  # "web" | "academic" | "books"
    final_output: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # "pending" | "running" | "completed" | "failed"
    error: Optional[str]
    top_of_mind_id: Optional[str]


class WrappedResearchState(TypedDict):
    """Complete state for wrapped research orchestration."""
    input: WrappedResearchInput

    # Workflow results
    web_result: Optional[WorkflowResult]
    academic_result: Optional[WorkflowResult]
    book_result: Optional[WorkflowResult]

    # Intermediate: book finding query generated from research
    book_theme: Optional[str]
    book_brief: Optional[str]

    # Final outputs
    combined_summary: Optional[str]
    top_of_mind_ids: dict[str, str]  # {workflow_type: uuid}

    # Checkpointing
    checkpoint_phase: CheckpointPhase
    checkpoint_path: Optional[str]

    # Metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_phase: str
    errors: Annotated[list[dict], add]  # Reducer for collecting errors
```

### Parallel Research Execution

Run web and academic in parallel with independent failure handling:

```python
# workflows/wrapped/nodes/run_parallel_research.py

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from workflows.research.graph.api import deep_research
from workflows.research.subgraphs.academic_lit_review import academic_lit_review
from workflows.wrapped.state import WrappedResearchState, WorkflowResult, QUALITY_MAPPING

logger = logging.getLogger(__name__)


async def run_parallel_research(state: WrappedResearchState) -> dict[str, Any]:
    """Run web and academic research workflows simultaneously.

    Uses asyncio.gather to run both in parallel. Each workflow is wrapped
    to capture errors independently - one failure doesn't prevent the other.
    """
    input_data = state["input"]
    quality_config = QUALITY_MAPPING[input_data["quality"]]

    async def run_web() -> WorkflowResult:
        """Run web research with error handling."""
        started_at = datetime.now(timezone.utc)
        try:
            logger.info(f"Starting web research: depth={quality_config['web_depth']}")
            result = await deep_research(
                query=input_data["query"],
                depth=quality_config["web_depth"],
            )
            return WorkflowResult(
                workflow_type="web",
                final_output=result.get("final_report"),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="completed",
                error=None,
                top_of_mind_id=None,
            )
        except Exception as e:
            logger.error(f"Web research failed: {e}")
            return WorkflowResult(
                workflow_type="web",
                final_output=None,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                error=str(e),
                top_of_mind_id=None,
            )

    async def run_academic() -> WorkflowResult:
        """Run academic lit review with error handling."""
        started_at = datetime.now(timezone.utc)

        # Generate research questions if not provided
        research_questions = input_data.get("research_questions") or [
            f"What are the main research themes in {input_data['query']}?",
            f"What methodological approaches are used?",
            f"What are the key findings and debates?",
        ]

        try:
            logger.info(f"Starting academic research: quality={quality_config['academic_quality']}")
            result = await academic_lit_review(
                topic=input_data["query"],
                research_questions=research_questions,
                quality=quality_config["academic_quality"],
                date_range=input_data.get("date_range"),
            )
            return WorkflowResult(
                workflow_type="academic",
                final_output=result.get("final_review"),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="completed",
                error=None,
                top_of_mind_id=None,
            )
        except Exception as e:
            logger.error(f"Academic research failed: {e}")
            return WorkflowResult(
                workflow_type="academic",
                final_output=None,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                error=str(e),
                top_of_mind_id=None,
            )

    # Run both in parallel - gather waits for both
    web_result, academic_result = await asyncio.gather(run_web(), run_academic())

    # Collect errors independently
    errors = []
    if web_result["status"] == "failed":
        errors.append({"phase": "web_research", "error": web_result["error"]})
    if academic_result["status"] == "failed":
        errors.append({"phase": "academic_research", "error": academic_result["error"]})

    logger.info(f"Parallel research complete. Web: {web_result['status']}, Academic: {academic_result['status']}")

    return {
        "web_result": web_result,
        "academic_result": academic_result,
        "current_phase": "parallel_research_complete",
        "errors": errors,
    }
```

### File-Based Checkpointing

Save state to disk after each major phase:

```python
# workflows/wrapped/checkpointing.py

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(".thala/checkpoints/wrapped")


def save_checkpoint(state: dict, prefix: str, phase: str) -> Path:
    """Save workflow state to checkpoint file.

    Args:
        state: Current workflow state (will be JSON serialized)
        prefix: Unique identifier for this run
        phase: Phase name (e.g., "after_parallel", "after_books")

    Returns:
        Path to saved checkpoint file
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{phase}_{timestamp}.json"
    checkpoint_path = CHECKPOINT_DIR / filename

    with open(checkpoint_path, "w") as f:
        json.dump(state, f, indent=2, default=str)

    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict | None:
    """Load workflow state from checkpoint file."""
    path = Path(path)
    if not path.exists():
        logger.error(f"Checkpoint not found: {path}")
        return None

    with open(path, "r") as f:
        state = json.load(f)

    logger.info(f"Checkpoint loaded: {path}")
    return state


def find_latest_checkpoint(prefix: str, phase: str | None = None) -> Path | None:
    """Find the most recent checkpoint for a run prefix.

    Args:
        prefix: Run identifier
        phase: Optional phase filter (e.g., "after_parallel")

    Returns:
        Path to most recent matching checkpoint, or None
    """
    if not CHECKPOINT_DIR.exists():
        return None

    pattern = f"{prefix}_{phase or '*'}_*.json"
    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)

    if checkpoints:
        return checkpoints[-1]
    return None
```

### Graph Construction

```python
# workflows/wrapped/graph/construction.py

from langgraph.graph import StateGraph, START, END

from workflows.wrapped.state import WrappedResearchState
from workflows.wrapped.nodes import (
    run_parallel_research,
    generate_book_query,
    run_book_finding,
    generate_final_summary,
    save_to_top_of_mind,
)
from workflows.wrapped.checkpointing import save_checkpoint


def create_wrapped_research_graph() -> StateGraph:
    """Create wrapped research workflow graph.

    Flow:
    START → parallel_research → generate_book_query → book_finding
          → final_summary → save_to_top_of_mind → END

    Checkpoints are saved after each major phase.
    """
    builder = StateGraph(WrappedResearchState)

    # Add nodes
    builder.add_node("parallel_research", run_parallel_research)
    builder.add_node("generate_book_query", generate_book_query)
    builder.add_node("book_finding", run_book_finding)
    builder.add_node("final_summary", generate_final_summary)
    builder.add_node("save_to_top_of_mind", save_to_top_of_mind)

    # Define flow
    builder.add_edge(START, "parallel_research")
    builder.add_edge("parallel_research", "generate_book_query")
    builder.add_edge("generate_book_query", "book_finding")
    builder.add_edge("book_finding", "final_summary")
    builder.add_edge("final_summary", "save_to_top_of_mind")
    builder.add_edge("save_to_top_of_mind", END)

    return builder.compile()


wrapped_research_graph = create_wrapped_research_graph()
```

### Public API

```python
# workflows/wrapped/graph/api.py

async def wrapped_research(
    query: str,
    quality: QualityTier = "standard",
    research_questions: list[str] | None = None,
    date_range: tuple[int, int] | None = None,
    checkpoint_prefix: str | None = None,
    resume_from: str | None = None,
) -> dict:
    """Run comprehensive research across web, academic, and book sources.

    Args:
        query: Research topic
        quality: Quality tier (quick, standard, comprehensive)
        research_questions: Optional specific questions for academic review
        date_range: Optional year range for academic papers
        checkpoint_prefix: Prefix for checkpoint files (enables checkpointing)
        resume_from: Path to checkpoint file to resume from

    Returns:
        Dict with web_result, academic_result, book_result, combined_summary

    Example:
        result = await wrapped_research(
            query="AI agents in creative work",
            quality="standard",
            checkpoint_prefix="ai_agents",
        )
    """
    if resume_from:
        state = load_checkpoint(resume_from)
        if not state:
            raise ValueError(f"Checkpoint not found: {resume_from}")
        # Continue from checkpoint...
    else:
        state = WrappedResearchState(
            input=WrappedResearchInput(
                query=query,
                quality=quality,
                research_questions=research_questions,
                date_range=date_range,
            ),
            web_result=None,
            academic_result=None,
            book_result=None,
            book_theme=None,
            book_brief=None,
            combined_summary=None,
            top_of_mind_ids={},
            checkpoint_phase=CheckpointPhase(
                parallel_research=False,
                book_query_generated=False,
                book_finding=False,
                saved_to_top_of_mind=False,
            ),
            checkpoint_path=None,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            current_phase="starting",
            errors=[],
        )

    result = await wrapped_research_graph.ainvoke(state)
    return result
```

## Usage

```python
from workflows.wrapped import wrapped_research

# Basic usage
result = await wrapped_research(
    query="AI agents in creative work",
    quality="standard",
)

# With checkpointing for long runs
result = await wrapped_research(
    query="Organizational resilience",
    quality="comprehensive",
    checkpoint_prefix="org_resilience",
)

# Resume from checkpoint
result = await wrapped_research(
    resume_from=".thala/checkpoints/wrapped/org_resilience_after_parallel_20260102_143022.json",
)

# Access results
print(result["web_result"]["final_output"][:500])
print(result["academic_result"]["final_output"][:500])
print(result["book_result"]["final_output"][:500])
print(result["combined_summary"])
```

## Guidelines

### Quality Mapping

When different workflows have different quality parameter names:
1. Define unified quality tiers (`quick`, `standard`, `comprehensive`)
2. Map to each workflow's specific parameter names
3. Document the mapping clearly

### Parallel Execution

For running workflows in parallel:
1. Use `asyncio.gather` for true parallelism
2. Wrap each workflow in try/except for independent failure handling
3. Collect errors separately - don't let one failure abort others
4. Return `WorkflowResult` objects with status tracking

### Checkpointing

For long-running workflows:
1. Save checkpoint after each major phase
2. Include all state needed to resume
3. Use `default=str` for datetime serialization
4. Provide both manual checkpoint path and auto-find by prefix
5. Track which phases completed in checkpoint state

### Error Handling

For multi-workflow orchestration:
1. Track errors per workflow independently
2. Use state reducers (`Annotated[list, add]`) for error collection
3. Continue with partial results when possible
4. Report which workflows succeeded/failed in final output

## Known Uses

- `workflows/wrapped/` - Full implementation orchestrating three research sources
- Test script with checkpoint support: `testing/test_wrapped_research.py`

## Consequences

### Benefits
- **Comprehensive research**: Multiple source types in single invocation
- **Parallelism**: Web and academic run simultaneously
- **Fault tolerance**: One workflow failing doesn't stop others
- **Resumability**: Checkpoint support for long runs
- **Unified API**: Single quality parameter configures all workflows

### Trade-offs
- **Complexity**: More state management than single workflow
- **Duration**: Comprehensive runs can take hours
- **Resource usage**: Multiple LLM calls, external APIs simultaneously

## Related Patterns

- [Standalone Book Finding Workflow](./standalone-book-finding-workflow.md) - Book source component
- [Citation Network Academic Review Workflow](./citation-network-academic-review-workflow.md) - Academic source component

## Related Solutions

- [Paper Acquisition Robustness](../../solutions/api-integration-issues/paper-acquisition-robustness.md) - Checkpoint patterns

## References

- [Python asyncio.gather](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
