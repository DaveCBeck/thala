---
name: incremental-checkpointing-iterative-workflows
title: "Incremental Checkpointing for Iterative Workflows"
date: 2026-01-30
category: data-pipeline
applicability:
  - "Long-running iterative operations with expensive per-iteration costs"
  - "Workflows that need mid-loop resumption after interruption"
  - "Supervision loops, batch processing, or any repeated expensive operations"
components: [checkpoint_manager, workflow_graph, async_task]
complexity: moderate
verified_in_production: true
related_solutions:
  - workflow-reliability/task-queue-interruption-recovery
tags: [checkpointing, resumption, graceful-shutdown, signal-handling, async, supervision-loops, gzip, atomic-writes]
---

# Incremental Checkpointing for Iterative Workflows

## Intent

Enable mid-iteration resumption for long-running iterative workflows by saving fine-grained progress checkpoints, complementing phase-level checkpoints with iteration-level granularity.

## Motivation

Phase-level checkpoints (e.g., "lit_review" → "supervision") are insufficient for iterative operations where each iteration is expensive. If a supervision loop runs 5 iterations at ~$1/iteration, losing 4 completed iterations means wasting $4 and time on restart.

**The Problem:**
```
Phase-level checkpoint: "supervision" started
├─ Iteration 1: ✓ ($1.00)
├─ Iteration 2: ✓ ($1.00)
├─ Iteration 3: ✓ ($1.00)
├─ Iteration 4: [SIGINT] ← Interruption
└─ Resume: Starts at iteration 1, not iteration 4!

Total waste: $3.00 + repeated time
```

**The Solution:**
```
Two-level checkpointing:

Phase Level (coarse)              Incremental Level (fine)
├─ "supervision" started          ├─ Iteration 1 checkpoint
│                                 ├─ Iteration 2 checkpoint
│                                 ├─ Iteration 3 checkpoint
│                                 ├─ [SIGINT]
│                                 └─ Resume: iteration 4
└─ "supervision" complete         └─ Clear incremental state

Waste: $0 (only incomplete iteration lost)
```

## Applicability

Use this pattern when:
- Iterations are expensive (LLM calls, API requests, heavy computation)
- Workflows run long enough that interruption is likely (> 5 minutes)
- State between iterations is serializable (JSON-compatible)
- Losing iteration progress has significant cost (time, money, rate limits)

Do NOT use this pattern when:
- Iterations are cheap (< 1 second, no API calls)
- State is not easily serializable (complex objects, file handles)
- Workflows complete quickly (< 1 minute total)

## Structure

```
.thala/queue/
├── queue.json              # Task definitions
├── current_work.json       # Phase-level checkpoints
└── incremental/            # Iteration-level checkpoints
    ├── {task_id_1}.json.gz
    ├── {task_id_2}.json.gz
    └── ...
```

**Component Diagram:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Task Queue Runner                                                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  run_task_workflow()                                          │   │
│  │  - Creates checkpoint_callback for phase transitions         │   │
│  │  - Creates IncrementalStateManager for mid-phase progress    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         ▼                    ▼                    ▼                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Checkpoint      │  │ Incremental     │  │ Workflow            │  │
│  │ Manager         │  │ State Manager   │  │ (with callback)     │  │
│  │ (phase-level)   │  │ (iteration)     │  │                     │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────┬───────────┘  │
│           │                    │                     │              │
│           ▼                    ▼                     ▼              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ current_work/   │  │ incremental/    │  │ Loop iterations     │  │
│  │ {task_id}.json  │  │ {task_id}.gz    │  │ call checkpoint_cb  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Participants

### IncrementalStateManager

Manages iteration-level checkpoints with gzip compression and atomic writes:

```python
class IncrementalStateManager:
    """Manages mid-phase incremental checkpoints."""

    def __init__(self, incremental_dir: Path | None = None):
        self._incremental_dir = incremental_dir or INCREMENTAL_DIR
        self._incremental_dir.mkdir(parents=True, exist_ok=True)

    async def save_progress(
        self,
        task_id: str,
        phase: str,
        iteration_count: int,
        partial_results: dict[str, Any],
        checkpoint_interval: int = 5,
    ) -> None:
        """Save incremental progress (async, uses thread pool)."""
        await asyncio.to_thread(
            self._save_progress_sync,
            task_id, phase, iteration_count, partial_results, checkpoint_interval
        )

    async def load_progress(
        self,
        task_id: str,
        phase: str | None = None,
    ) -> IncrementalState | None:
        """Load incremental progress for resumption."""
        return await asyncio.to_thread(self._load_progress_sync, task_id, phase)

    async def clear_progress(self, task_id: str) -> bool:
        """Clear when phase completes successfully."""
        return await asyncio.to_thread(self._clear_progress_sync, task_id)
```

### IncrementalState TypedDict

Schema for checkpoint data:

```python
class IncrementalState(TypedDict):
    task_id: str
    phase: str
    iteration_count: int
    checkpoint_interval: int
    partial_results: dict[str, Any]  # Application-specific
    last_checkpoint_at: str  # ISO timestamp
```

### Checkpoint Callback Type

Signature for callbacks passed to iterative operations:

```python
IncrementalCheckpointCallback = Callable[[int, dict], Awaitable[None]]
"""
Args:
    iteration: Current iteration number (1-indexed)
    partial_results: Dict containing state to preserve
"""
```

## Collaborations

### Workflow Integration Flow

```
1. Runner creates IncrementalStateManager
2. Runner loads incremental state if resuming
3. Runner creates checkpoint callback that calls manager.save_progress()
4. Workflow receives callback and incremental_state parameters
5. Workflow uses incremental_state to skip completed iterations
6. Workflow calls callback after each iteration
7. Manager saves gzip-compressed JSON atomically
8. On phase completion, runner calls manager.clear_progress()
```

### Resume Flow

```
1. Runner detects incomplete work via CheckpointManager
2. Runner loads phase-level checkpoint (current phase, phase_outputs)
3. Runner loads incremental state for current phase
4. Workflow receives both: skip completed phases + resume mid-iteration
5. Workflow uses incremental_state.iteration_count to start from correct point
6. Workflow uses incremental_state.partial_results to restore state
```

## Implementation

### Basic Iterative Operation with Checkpointing

```python
async def expensive_batch_operation(
    items: list[str],
    checkpoint_callback: IncrementalCheckpointCallback | None = None,
    incremental_state: IncrementalState | None = None,
) -> dict[str, Any]:
    """Process items with checkpoint support."""

    # Resume from checkpoint if available
    results = {}
    start_index = 0

    if incremental_state:
        results = incremental_state.get("partial_results", {})
        start_index = incremental_state.get("iteration_count", 0)
        logger.info(f"Resuming from item {start_index}, {len(results)} already done")

    # Process remaining items
    for i, item in enumerate(items[start_index:], start=start_index):
        result = await expensive_api_call(item)
        results[item] = result

        # Checkpoint every 5 items
        if checkpoint_callback and (i + 1) % 5 == 0:
            await checkpoint_callback(i + 1, {"results": results})

    return results
```

### Supervision Loop with Checkpointing

```python
async def run_supervision_loop(
    review: str,
    max_iterations: int = 5,
    checkpoint_callback: IncrementalCheckpointCallback | None = None,
    incremental_state: IncrementalState | None = None,
) -> str:
    """Supervision loop with per-iteration checkpointing."""

    # Resume state
    current_review = review
    start_iteration = 1

    if incremental_state:
        partial = incremental_state.get("partial_results", {})
        current_review = partial.get("current_review", review)
        start_iteration = incremental_state.get("iteration_count", 0) + 1
        logger.info(f"Resuming supervision from iteration {start_iteration}")

    # Run iterations
    for iteration in range(start_iteration, max_iterations + 1):
        # Expensive operation (e.g., Opus call)
        issues = await identify_issues(current_review)
        current_review = await integrate_improvements(current_review, issues)

        # Checkpoint after EVERY iteration (expensive operations)
        if checkpoint_callback:
            await checkpoint_callback(
                iteration,
                {
                    "current_review": current_review,
                    "iteration": iteration,
                }
            )

    return current_review
```

### Wiring Through Workflow Chain

```python
# In task queue runner

async def run_task_workflow(task: Task, resume_from: WorkflowCheckpoint | None):
    task_id = task["id"]
    incremental_mgr = IncrementalStateManager()

    # Load incremental state if resuming
    incremental_state = None
    if resume_from:
        current_phase = resume_from.get("phase", "")
        incremental_state = await incremental_mgr.load_progress(task_id, current_phase)

    # Create callback for supervision phase
    async def supervision_checkpoint(iteration: int, partial_results: dict) -> None:
        await incremental_mgr.save_progress(
            task_id=task_id,
            phase="supervision",
            iteration_count=iteration,
            partial_results=partial_results,
            checkpoint_interval=1,  # Every iteration for expensive ops
        )

    # Run workflow with callback
    result = await workflow.run(
        task,
        checkpoint_callback=supervision_checkpoint,
        incremental_state=incremental_state,
    )

    # Clear incremental state on success
    await incremental_mgr.clear_progress(task_id)

    return result
```

### Atomic Writes with Gzip Compression

```python
def _save_progress_sync(self, task_id: str, phase: str, ...):
    """Sync implementation for atomic gzip writes."""
    checkpoint_path = self._incremental_dir / f"{task_id}.json.gz"
    temp_path = checkpoint_path.with_suffix(".tmp.gz")

    state: IncrementalState = {
        "task_id": task_id,
        "phase": phase,
        "iteration_count": iteration_count,
        "checkpoint_interval": checkpoint_interval,
        "partial_results": partial_results,
        "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write to temp file first
    with gzip.open(temp_path, "wt", encoding="utf-8") as f:
        json.dump(state, f)

    # Atomic rename (POSIX guarantees)
    temp_path.rename(checkpoint_path)
```

## Consequences

### Benefits

1. **Cost savings**: Only lose progress from incomplete iteration, not entire phase
2. **Graceful resume**: Workflows continue from exact interruption point
3. **Low overhead**: Gzip compression reduces checkpoint size 10-30x
4. **Non-blocking**: Async I/O via `asyncio.to_thread()` keeps event loop responsive
5. **Corruption-safe**: Atomic writes prevent partial checkpoint files

### Liabilities

1. **Additional complexity**: Two checkpoint levels to manage
2. **Storage overhead**: More frequent writes (mitigated by compression)
3. **Callback threading**: Must pass callbacks through multiple layers
4. **State schema**: partial_results must be JSON-serializable

### Trade-offs

| Checkpoint Interval | Storage Overhead | Resume Granularity | Write Frequency |
|---------------------|------------------|-------------------|-----------------|
| Every iteration     | Higher           | Best              | High            |
| Every 5 iterations  | Medium           | Good              | Medium          |
| Every 10 iterations | Lower            | Acceptable        | Low             |

**Recommendation**: Use interval=1 for expensive operations (>$0.10/iteration), interval=5 for medium cost.

## Known Uses

### Supervision Loops (Loop1, Loop2)

Each iteration involves expensive Opus calls for issue identification and integration:

```python
# workflows/enhance/supervision/loop1/graph.py
checkpoint_callback(
    iteration,
    {
        "current_review": integrated_review,
        "iteration": iteration,
        "explored_bases": explored_bases,
    }
)
```

### Paper Processing Pipeline

Batch processing with checkpoints every 5 papers:

```python
# workflows/research/academic_lit_review/paper_processor/nodes.py
if checkpoint_callback and (i + 1) % 5 == 0:
    await checkpoint_callback(i + 1, {"processing_results": results})
```

### Delta-Based Supervision State

For supervision loops with large corpus state, store only deltas:

```python
# Instead of full corpus (~10MB), store references
checkpoint_callback(
    iteration,
    {
        "current_review": updated_review,  # The text itself
        "iteration": iteration,
        "new_dois": newly_added_dois,       # Just the DOI list (~1KB)
        # Full corpus reconstructed from phase_outputs + ES queries
    }
)
```

## Related Patterns

- [Task Queue with Budget Tracking](./task-queue-budget-tracking.md) - Phase-level checkpointing this extends
- [Multi-Workflow Task Queue](./multi-workflow-task-queue.md) - Workflow-aware checkpoint phases
- [Workflow State Decoupling](../langgraph/workflow-state-decoupling.md) - Minimal state via store queries

## Related Solutions

- [Task Queue Interruption Recovery](../../solutions/workflow-reliability/task-queue-interruption-recovery.md) - Full implementation details
