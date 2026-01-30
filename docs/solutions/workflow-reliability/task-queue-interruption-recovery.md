---
module: core/task_queue, workflows/enhance/supervision
date: 2026-01-30
problem_type: data_loss, reliability
component: task_queue, checkpoint_manager, shutdown_coordinator
symptoms:
  - "Long-running tasks lose all progress on SIGINT/SIGTERM"
  - "Supervision loops restart from iteration 0 after system crash"
  - "Expensive Opus calls repeated on every resume"
  - "Ctrl+C kills process without cleanup"
  - "No way to resume mid-phase within iterative operations"
root_cause: missing_incremental_checkpoints, blocking_signal_handling
resolution_type: feature_implementation
severity: high
tags: [task-queue, checkpointing, graceful-shutdown, signal-handling, resumption, async, supervision-loops]
---

# Task Queue Interruption and Recovery

## Problem

Long-running workflow tasks (academic literature reviews, web research, supervision loops) would lose all progress if interrupted by a system crash, signal, or timeout. Users had to restart from scratch, wasting compute time and API costs.

### Environment

- **Module**: `core/task_queue`, `workflows/enhance/supervision`
- **Python**: 3.12
- **Affected workflows**: `lit_review_full`, supervision loops (Loop1, Loop2)

### Symptoms

**Progress Loss:**
```
[INFO] Supervision Loop 1: iteration 4/5
[SIGINT received]
[Process killed]

# On next run:
[INFO] Resuming task abc123
[INFO] Supervision Loop 1: iteration 1/5  ← Lost 4 iterations of Opus calls
```

**No Graceful Shutdown:**
```
# User presses Ctrl+C during 5-minute idle wait
[Process killed immediately]
# No cleanup, no checkpoint save, orphaned temp files remain
```

**Expensive Operations Repeated:**
```
# Paper acquisition: 50 papers processed
[SIGINT received]
# On resume: Starts from paper 1, not paper 50
# Cost: Additional ~$15 in API calls for papers already processed
```

## Investigation

### What Didn't Work

1. **Phase-level checkpoints only**
   - Why it failed: Checkpoints only saved at phase boundaries (e.g., "lit_review" → "supervision"). Interruption mid-supervision-loop lost all iteration progress.

2. **Blocking signal handlers**
   - Why it failed: `signal.signal(SIGINT, handler)` blocks the asyncio event loop. Long sleeps (5 min poll intervals) couldn't be interrupted.

3. **Simple process restart**
   - Why it failed: No state saved within iterative phases. Resumption restarted entire phases, repeating expensive operations.

## Root Cause

### Issue 1: No Mid-Phase Checkpointing

Checkpoints only tracked completed phases, not iteration progress within phases:

```python
# PROBLEMATIC: Only phase-level checkpoints
checkpoint = {
    "phase": "supervision",  # Which phase we're in
    "phase_outputs": {...},   # Outputs from completed phases
    # No iteration progress within "supervision"!
}
```

Supervision loops run 3-5 iterations with expensive Opus calls each. Losing mid-loop progress meant repeating $2-5 in API calls per restart.

### Issue 2: Blocking Signal Handling

Signal handlers used sync patterns that blocked the event loop:

```python
# PROBLEMATIC: Blocks event loop
signal.signal(signal.SIGINT, lambda s, f: shutdown_flag.set())

# Long sleep can't be interrupted
await asyncio.sleep(300)  # 5 min wait, SIGINT waits for completion
```

### Issue 3: No Atomic State Persistence

Checkpoint writes could be interrupted mid-write, corrupting state files:

```python
# PROBLEMATIC: Non-atomic write
with open(checkpoint_path, 'w') as f:
    json.dump(state, f)  # Interrupted = corrupt JSON
```

## Solution

### Architecture: Two-Level Checkpointing

```
Workflow Level (coarse)          Incremental Level (fine)
├─ Phase Start                   ├─ Iteration 1-4
├─ Checkpoint Phase              ├─ Checkpoint (5 papers)
├─ Resume Phase                  ├─ Iteration 5-9
└─ Phase Complete                ├─ Checkpoint (10 papers)
                                 └─ ...on resume, skip to iteration 10
```

### Component 1: IncrementalStateManager

Saves fine-grained progress within iterative operations:

```python
# core/task_queue/incremental_state.py

class IncrementalStateManager:
    """Manages mid-phase incremental checkpoints with gzip compression."""

    async def save_progress(
        self,
        task_id: str,
        phase: str,
        iteration_count: int,
        partial_results: dict[str, Any],
        checkpoint_interval: int = 5,
    ) -> None:
        """Save incremental progress for a task.

        Uses atomic writes (temp + rename) and gzip compression (~10-30x).
        All I/O offloaded to thread pool via asyncio.to_thread().
        """
        await asyncio.to_thread(
            self._save_progress_sync,
            task_id, phase, iteration_count, partial_results, checkpoint_interval
        )

    async def load_progress(
        self,
        task_id: str,
        phase: Optional[str] = None,
    ) -> Optional[IncrementalState]:
        """Load incremental progress, optionally filtered by phase."""
        return await asyncio.to_thread(self._load_progress_sync, task_id, phase)

    async def clear_progress(self, task_id: str) -> bool:
        """Clear incremental progress when phase completes."""
        return await asyncio.to_thread(self._clear_progress_sync, task_id)
```

**Storage location**: `.thala/queue/incremental/{task_id}.json.gz`

**Usage in supervision loops:**

```python
# workflows/enhance/supervision/loop1/graph.py

async def run_loop1_standalone(
    review: str,
    max_iterations: int = 3,
    checkpoint_callback: Callable[[int, dict], None] | None = None,
    incremental_state: Optional[IncrementalState] = None,
) -> Loop1Output:
    """Run Loop 1 with incremental checkpointing for mid-loop resumption."""

    # Handle resume from incremental state
    start_iteration = 1
    current_review = review
    explored_bases = []

    if incremental_state:
        partial = incremental_state.get("partial_results", {})
        start_iteration = incremental_state.get("iteration_count", 0) + 1
        current_review = partial.get("current_review", review)
        explored_bases = partial.get("explored_bases", [])
        logger.info(f"Resuming Loop 1 from iteration {start_iteration}")

    # Run iterations with checkpointing
    for iteration in range(start_iteration, max_iterations + 1):
        # Process iteration (expensive Opus calls)
        issues = await identify_theoretical_issues(current_review)
        current_review = await deepen_analysis(current_review, issues)

        # Checkpoint after each iteration
        if checkpoint_callback:
            await checkpoint_callback(
                iteration,
                {
                    "current_review": current_review,
                    "iteration": iteration,
                    "explored_bases": explored_bases,
                }
            )

    return Loop1Output(current_review=current_review, ...)
```

### Component 2: ShutdownCoordinator

Asyncio-native signal handling with immediate wake on shutdown:

```python
# core/task_queue/shutdown.py

class ShutdownCoordinator:
    """Manages graceful shutdown with immediate interruption of long waits."""

    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def request_shutdown(self) -> None:
        """Request graceful shutdown (idempotent)."""
        if not self._shutdown_event.is_set():
            logger.info("Shutdown signal received")
            self._shutdown_event.set()

    async def wait_or_shutdown(self, timeout: float) -> bool:
        """Wait for timeout or shutdown signal.

        Returns True if shutdown was requested, False if timeout expired.
        Replaces asyncio.sleep() for interruptible waits.
        """
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
            return True  # Shutdown requested
        except asyncio.TimeoutError:
            return False  # Normal timeout

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers using asyncio-native approach."""
        self._loop = asyncio.get_running_loop()
        self._loop.add_signal_handler(signal.SIGINT, self.request_shutdown)
        self._loop.add_signal_handler(signal.SIGTERM, self.request_shutdown)
```

**Usage in runner:**

```python
# core/task_queue/runner.py

async def run_queue_loop(...):
    coordinator = get_shutdown_coordinator()
    coordinator.install_signal_handlers()

    while not coordinator.shutdown_requested:
        # Process tasks...

        # Interruptible wait (replaces asyncio.sleep)
        if await coordinator.wait_or_shutdown(300):  # 5 min
            logger.info("Shutdown requested, exiting")
            break
```

### Component 3: Atomic Writes with Compression

Prevent corruption from interrupted writes:

```python
# In IncrementalStateManager._save_progress_sync()

def _save_progress_sync(self, task_id: str, phase: str, ...):
    checkpoint_path = self._incremental_dir / f"{task_id}.json.gz"
    temp_path = checkpoint_path.with_suffix(".tmp.gz")

    state = IncrementalState(
        task_id=task_id,
        phase=phase,
        iteration_count=iteration_count,
        partial_results=partial_results,
        ...
    )

    # Atomic write: temp file + rename
    with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
        json.dump(state, f)

    temp_path.rename(checkpoint_path)  # Atomic on POSIX
```

**Orphaned temp cleanup at startup:**

```python
async def cleanup_orphaned_temps(self) -> int:
    """Clean up .tmp.gz files from interrupted writes."""
    count = 0
    for tmp_file in self._incremental_dir.glob("*.tmp.gz"):
        tmp_file.unlink()
        count += 1
    return count
```

### Workflow Integration

Wiring checkpoint callbacks through the full chain:

```python
# core/task_queue/workflows/lit_review_full.py

async def run(self, task: dict, checkpoint_callback, resume_from=None):
    task_id = task["id"]
    incremental_mgr = IncrementalStateManager()

    # Load incremental state if resuming
    incremental_state = None
    if resume_from:
        current_phase = resume_from.get("phase", "")
        incremental_state = await incremental_mgr.load_progress(task_id, current_phase)

    # Create checkpoint callback for supervision
    async def supervision_checkpoint(iteration: int, partial_results: dict) -> None:
        await incremental_mgr.save_progress(
            task_id=task_id,
            phase="supervision",
            iteration_count=iteration,
            partial_results=partial_results,
            checkpoint_interval=1,  # Every iteration for expensive ops
        )

    # Pass to supervision API
    enhance_result = await enhance_report(
        report=lit_result["final_review"],
        checkpoint_callback=supervision_checkpoint,
        incremental_state=incremental_state,
    )

    # Clear incremental state when phase completes
    await incremental_mgr.clear_progress(task_id)
```

### Files Modified

**New files:**
- `core/task_queue/incremental_state.py`: IncrementalStateManager (330 lines)
- `core/task_queue/shutdown.py`: ShutdownCoordinator (147 lines)
- `testing/test_task_queue_interruption.py`: Test suite (262 lines)

**Modified files:**
- `core/task_queue/runner.py`: Integrated shutdown coordinator and checkpoint callbacks
- `core/task_queue/checkpoint_manager.py`: Made async with `asyncio.to_thread()`
- `core/task_queue/schemas.py`: Added `IncrementalState` TypedDict, callback type aliases
- `core/task_queue/paths.py`: Added `INCREMENTAL_DIR` constant
- `core/task_queue/workflows/lit_review_full.py`: Integrated incremental state management
- `workflows/enhance/__init__.py`: Added `checkpoint_callback` parameter
- `workflows/enhance/supervision/api.py`: Wired callbacks to loops
- `workflows/enhance/supervision/loop1/graph.py`: Added mid-loop checkpointing
- `workflows/enhance/supervision/loop2/graph.py`: Added mid-loop checkpointing

## Prevention

### How to Avoid This

1. **Always use `wait_or_shutdown()` instead of `asyncio.sleep()`**
   ```python
   # Bad: Uninterruptible
   await asyncio.sleep(300)

   # Good: Interruptible
   if await coordinator.wait_or_shutdown(300):
       return  # Shutdown requested
   ```

2. **Add checkpoint callbacks to iterative operations**
   ```python
   # Any loop with expensive operations should accept a callback
   async def expensive_loop(
       items: list,
       checkpoint_callback: Callable[[int, dict], None] | None = None,
   ):
       for i, item in enumerate(items):
           result = await expensive_operation(item)
           if checkpoint_callback and (i + 1) % 5 == 0:
               await checkpoint_callback(i + 1, {"results": results})
   ```

3. **Use atomic writes for state persistence**
   ```python
   # Write to temp, then rename
   temp_path = path.with_suffix('.tmp')
   with open(temp_path, 'w') as f:
       json.dump(data, f)
   temp_path.rename(path)  # Atomic on POSIX
   ```

4. **Use `asyncio.to_thread()` for file I/O in async code**
   ```python
   # Bad: Blocks event loop
   with open(path, 'w') as f:
       json.dump(data, f)

   # Good: Offloads to thread pool
   await asyncio.to_thread(self._save_sync, path, data)
   ```

5. **Use `loop.add_signal_handler()` for async signal handling**
   ```python
   # Bad: Blocks
   signal.signal(signal.SIGINT, handler)

   # Good: Asyncio-native
   loop.add_signal_handler(signal.SIGINT, coordinator.request_shutdown)
   ```

### Test Cases

```python
async def test_shutdown_coordinator_wait_or_shutdown():
    """Test shutdown coordinator's interruptible wait."""
    coordinator = ShutdownCoordinator()

    # Normal timeout
    result = await coordinator.wait_or_shutdown(0.1)
    assert result is False

    # Shutdown interrupts wait immediately
    asyncio.create_task(coordinator.request_shutdown())
    start = time.time()
    result = await coordinator.wait_or_shutdown(10.0)
    elapsed = time.time() - start

    assert result is True  # Shutdown requested
    assert elapsed < 1.0   # Returned immediately, not after 10s


async def test_incremental_checkpoint_resume():
    """Test mid-loop resumption from checkpoint."""
    manager = IncrementalStateManager()

    # Simulate interruption at iteration 3
    await manager.save_progress(
        task_id="test-123",
        phase="supervision",
        iteration_count=3,
        partial_results={"current_review": "partial text", "iteration": 3},
    )

    # Resume should continue from iteration 4
    state = await manager.load_progress("test-123", "supervision")
    assert state["iteration_count"] == 3
    start_iteration = state["iteration_count"] + 1  # = 4
    assert start_iteration == 4
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| Resume granularity | Phase-level | Iteration-level |
| SIGINT response | After current sleep | Immediate |
| Supervision restart cost | $2-5 (full loop) | $0.50-1 (partial) |
| Checkpoint corruption | Possible | Prevented (atomic) |
| File I/O blocking | Yes | No (async) |
| Checkpoint size (supervision) | ~10MB | ~200KB (gzip) |

## Related

- [Multi-Workflow Task Queue Architecture](../../patterns/data-pipeline/multi-workflow-task-queue.md) - Registry-based workflow dispatch
- [Task Queue with Budget Tracking](../../patterns/data-pipeline/task-queue-budget-tracking.md) - Budget thresholds and cost tracking
- [Workflow State Decoupling](../../patterns/langgraph/workflow-state-decoupling.md) - Minimal state via direct store queries
- [Workflow State and Truncation Fixes](./workflow-state-truncation-fixes.md) - State loading patterns
