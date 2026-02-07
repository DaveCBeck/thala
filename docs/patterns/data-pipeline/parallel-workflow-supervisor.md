---
name: parallel-workflow-supervisor
title: "Parallel Workflow Supervisor: Concurrent Task Execution with Rate Limiting and Shared Resource Management"
date: 2026-02-07
category: data-pipeline
applicability:
  - "Running multiple long-duration LLM workflows concurrently"
  - "Bypassing sequential queue processing for throughput"
  - "Rate-limiting shared external APIs during parallel execution"
  - "Managing singleton resource lifecycle across concurrent workflows"
components: [async_task, workflow_graph, configuration]
complexity: complex
verified_in_production: false
related_solutions:
  - "async-issues/sequential-to-parallel-asyncio-gather"
  - "async-issues/http-client-cleanup-registry"
related_patterns:
  - "data-pipeline/multi-workflow-task-queue"
  - "data-pipeline/task-queue-budget-tracking"
tags: [parallel, asyncio-gather, semaphore, rate-limiting, supervisor, stagger, singleton, broker-lifecycle, task-queue, concurrency]
---

# Parallel Workflow Supervisor: Concurrent Task Execution with Rate Limiting and Shared Resource Management

## Intent

Provide a supervisor that runs N task workflows concurrently via `asyncio.gather()`, with staggered starts, per-API rate limiting via semaphores, and centralized resource lifecycle management (LLM broker, HTTP clients).

## Motivation

The existing sequential queue loop (`queue_loop.py`) processes one task at a time. Literature reviews take 15-30 minutes each. Running 5 tasks sequentially takes hours; running them concurrently with staggered starts reduces wall-clock time dramatically while respecting API rate limits.

**The Problem:**
```
Sequential Queue Loop
Task 1: [=============================] 20min
Task 2:                               [=============================] 20min
Task 3:                                                             [=====...
Total time: 100+ minutes for 5 tasks
```

**The Solution:**
```
Parallel Supervisor with Staggered Starts
Task 1: [=============================] 20min
Task 2:    [=============================] 20min (3min stagger)
Task 3:       [=============================] 20min (6min stagger)
Task 4:          [=============================] 20min (9min stagger)
Task 5:             [=============================] 20min (12min stagger)
Total time: ~30 minutes for 5 tasks (3x faster)

+ Per-API rate limiting via semaphores
+ Centralized broker lifecycle (no premature shutdown)
+ Orphaned task recovery on crash
```

### Key Challenges When Moving from Sequential to Parallel

1. **Shared singleton resources**: The LLM broker is a singleton — individual workflows must NOT stop it when they complete, or it kills in-flight workflows
2. **API rate limits**: External APIs (Imagen, OpenAlex) have concurrency limits that must be enforced across all parallel workflows
3. **Stale semaphores**: Module-level `asyncio.Semaphore` objects go stale across `asyncio.run()` boundaries — lazy factory pattern needed
4. **Sync I/O in async context**: File-based queue uses `fcntl.flock()` which blocks the event loop — needs `asyncio.to_thread()`
5. **Orphaned tasks**: If the supervisor crashes or gets cancelled, IN_PROGRESS tasks must be reset to PENDING

## Applicability

Use this pattern when:
- Multiple long-duration workflows can run concurrently (no data dependencies)
- External APIs have per-API concurrency limits (not global)
- Shared singleton resources (broker, HTTP clients) need lifecycle management
- Staggered starts help prevent API thundering herd
- Tasks can be atomically selected and claimed from a queue

Do NOT use this pattern when:
- Workflows have strict ordering dependencies
- API rate limits are global (single semaphore across all APIs)
- Sequential processing is sufficient (throughput not bottlenecked)
- Resource requirements exceed system capacity (memory, CPU)

## Structure

```
Parallel Workflow Supervisor Architecture

+-----------------------------------------------------------------------------+
|  parallel.py (Supervisor)                                                   |
|  +-----------------------------------------------------------------------+  |
|  | run_parallel_tasks(count, stagger_minutes)                            |  |
|  |   1. Atomic task selection: _select_tasks() via asyncio.to_thread()  |  |
|  |   2. Stagger starts: run_with_stagger() with coordinator.wait()      |  |
|  |   3. Concurrent execution: asyncio.gather(return_exceptions=True)    |  |
|  |   4. Finally: cleanup_supervisor_resources(), reset orphaned tasks   |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
          |                                   |
          v                                   v
+--------------------+            +-------------------------------+
| rate_limits.py     |            | lifecycle.py                  |
| - Lazy factories   |            | - cleanup_supervisor_resources|
|                    |            |   - broker.stop()             |
| get_imagen_sem()   |            |   - cleanup_all_clients()     |
| get_openalex_sem() |            +-------------------------------+
+--------------------+
          |
          v
+-----------------------------------------------------------------------------+
| Semaphore Consumers                                                         |
|                                                                             |
| image_utils.py:                    openalex/client.py:                      |
|   async with get_imagen_semaphore():   class _SemaphoreClient:             |
|     client.models.generate_images()      async with get_openalex_sem():    |
|                                            return await client.get(...)     |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| workflow_executor.py                                                        |
| - run_task_workflow(task, ...) — runs individual workflows                 |
| - Does NOT stop broker (lifecycle managed by supervisor)                   |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| commands/parallel_command.py (CLI Entry)                                   |
| - thala-queue parallel --count 5 --stagger 3.0                             |
| - Auto-scales broker: THALA_LLM_BROKER_MAX_CONCURRENT_SYNC = count * 3     |
+-----------------------------------------------------------------------------+
```

## Implementation

### Step 1: Atomic Task Selection with File Locking

```python
# core/task_queue/parallel.py

async def run_parallel_tasks(
    count: int = 5,
    stagger_minutes: float = 3.0,
    queue_dir: Path | None = None,
) -> list[dict | BaseException]:
    """Run multiple tasks concurrently via asyncio.gather().

    Bypasses the queue loop for parallel execution. Manages broker
    lifecycle at supervisor level.
    """
    coordinator = get_shutdown_coordinator()
    coordinator.install_signal_handlers()
    queue_manager = TaskQueueManager(queue_dir=queue_dir)
    checkpoint_mgr = CheckpointManager(queue_dir=queue_dir)
    budget_tracker = BudgetTracker(queue_dir=queue_dir)

    # Atomic task selection (sync I/O with fcntl.flock — run in thread pool)
    tasks = await asyncio.to_thread(_select_tasks, queue_manager, count)
    if not tasks:
        logger.info("No eligible tasks to run")
        coordinator.remove_signal_handlers()
        return []

    logger.info(f"Selected {len(tasks)} tasks for parallel execution")
    selected_ids = {task["id"] for task in tasks}
    # ... continued below


def _select_tasks(queue_manager: TaskQueueManager, count: int) -> list[Task]:
    """Atomically select and claim pending tasks under queue lock.

    Called via asyncio.to_thread() to avoid blocking the event loop
    with fcntl.flock().

    Tasks are sorted by priority (highest first), then by creation time
    (oldest first) for FIFO within the same priority level.
    """
    with queue_manager.persistence.lock():
        queue = queue_manager.persistence.read_queue()
        pending = [t for t in queue["topics"] if t["status"] == "pending"]
        # Sort by priority descending (4=urgent > 3=high > 2=normal > 1=low),
        # then by created_at ascending (FIFO within same priority).
        pending.sort(key=lambda t: (-t.get("priority", 2), t.get("created_at", "")))
        selected = pending[:count]
        now = datetime.now(timezone.utc).isoformat()
        for task in selected:
            task["status"] = "in_progress"
            task["started_at"] = now
        queue_manager.persistence.write_queue(queue)
    return selected
```

**Key Details:**
- `asyncio.to_thread()` prevents `fcntl.flock()` from blocking the event loop
- Priority-first, then FIFO ordering matches sequential queue behavior
- Tasks are atomically claimed (status=IN_PROGRESS) under file lock
- Returns empty list if no eligible tasks (caller can exit early)

### Step 2: Staggered Concurrent Execution

```python
# core/task_queue/parallel.py (continued)

    try:
        # Stagger starts and run concurrently
        async def run_with_stagger(task: Task, index: int) -> dict:
            if index > 0:
                delay = index * stagger_minutes * 60
                tid = task["id"][:8]
                logger.info(f"Task {tid}: starting in {delay:.0f}s")
                if await coordinator.wait_or_shutdown(delay):
                    logger.info(f"Task {tid}: skipped due to shutdown")
                    raise asyncio.CancelledError()

            # Check budget before launching workflow
            should_proceed, reason = budget_tracker.should_proceed()
            if not should_proceed:
                tid = task["id"][:8]
                logger.warning(f"Task {tid}: skipped due to budget: {reason}")
                return {"status": "skipped", "reason": f"budget: {reason}"}

            return await run_task_workflow(
                task,
                queue_manager,
                checkpoint_mgr,
                budget_tracker,
                shutdown_coordinator=coordinator,
            )

        results = await asyncio.gather(
            *[run_with_stagger(task, i) for i, task in enumerate(tasks)],
            return_exceptions=True,
        )

        # Log results
        for task, result in zip(tasks, results):
            tid = task["id"][:8]
            if isinstance(result, asyncio.CancelledError):
                logger.info(f"Task {tid} was cancelled")
            elif isinstance(result, BaseException):
                logger.error(f"Task {tid} failed: {result}")
            else:
                logger.info(f"Task {tid}: {result.get('status', 'unknown')}")

        return results
```

**Key Details:**
- Index 0 starts immediately, others wait stagger_minutes * index
- `coordinator.wait_or_shutdown()` allows early exit on SIGINT/SIGTERM
- Budget check per task (not global) for finer control
- `return_exceptions=True` prevents one failure from cancelling others
- Caller must check if each result is BaseException or dict

### Step 3: Lazy Semaphore Factory (Avoids Stale Semaphores)

```python
# core/task_queue/rate_limits.py

"""Global rate-limit semaphores for parallel workflow execution.

Uses lazy factory functions to avoid stale semaphores when asyncio.run()
is called multiple times (e.g. in tests). Each semaphore is created on
first access within the current event loop.
"""

import asyncio
import os

_imagen_semaphore: asyncio.Semaphore | None = None
_openalex_semaphore: asyncio.Semaphore | None = None


def get_imagen_semaphore() -> asyncio.Semaphore:
    """Get or create the global Imagen API semaphore."""
    global _imagen_semaphore
    if _imagen_semaphore is None:
        limit = int(os.environ.get("THALA_IMAGEN_CONCURRENCY", "10"))
        _imagen_semaphore = asyncio.Semaphore(limit)
    return _imagen_semaphore


def get_openalex_semaphore() -> asyncio.Semaphore:
    """Get or create the global OpenAlex API semaphore."""
    global _openalex_semaphore
    if _openalex_semaphore is None:
        limit = int(os.environ.get("THALA_OPENALEX_CONCURRENCY", "20"))
        _openalex_semaphore = asyncio.Semaphore(limit)
    return _openalex_semaphore
```

**Key Details:**
- Module-level `asyncio.Semaphore` goes stale across `asyncio.run()` calls
- Lazy factory pattern creates semaphore on first access in current loop
- Env vars allow runtime configuration per API
- Independent semaphores per API (not global)

### Step 4: Semaphore-Wrapped HTTP Client

```python
# langchain_tools/openalex/client.py

class _SemaphoreClient:
    """Thin wrapper around httpx.AsyncClient that applies a semaphore to .get() calls."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def get(self, *args: Any, **kwargs: Any) -> httpx.Response:
        from core.task_queue.rate_limits import get_openalex_semaphore

        async with get_openalex_semaphore():
            return await self._client.get(*args, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()


def _get_openalex() -> _SemaphoreClient:
    """Get OpenAlex httpx client (lazy init).

    Returns a thin wrapper that applies a global semaphore to .get() calls,
    limiting concurrent OpenAlex requests during parallel workflow execution.
    """
    global _openalex_client
    if _openalex_client is None:
        # ... setup raw httpx.AsyncClient ...
        _openalex_client = _SemaphoreClient(raw_client)
        register_cleanup("OpenAlex", close_openalex)
    return _openalex_client
```

**Key Details:**
- Semaphore applied at HTTP client level (transparent to callers)
- Lazy import of `get_openalex_semaphore()` avoids circular deps
- All `.get()` calls are automatically rate-limited
- Registered for cleanup via HTTP client registry

### Step 5: Semaphore Usage in Imagen API

```python
# workflows/shared/image_utils.py

async def generate_article_header(
    title: str,
    content: str,
    custom_prompt: str | None = None,
    aspect_ratio: str = "16:9",
) -> tuple[bytes | None, str | None]:
    """Generate article header image using LLM-generated prompt + Imagen."""
    from google.genai import types

    client = _get_genai_client()

    # Step 1: Use custom prompt or generate one using Sonnet
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = await generate_image_prompt(title, content)
        if not prompt:
            return None, None

    # Step 2: Generate the image (semaphore limits concurrent API calls)
    from core.task_queue.rate_limits import get_imagen_semaphore

    async with get_imagen_semaphore():
        response = await client.aio.models.generate_images(
            model=IMAGEN_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            ),
        )

    if response.generated_images:
        image_bytes = response.generated_images[0].image.image_bytes
        return image_bytes, prompt

    return None, prompt
```

**Key Details:**
- Single entry point for Imagen API calls across all workflows
- Semaphore enforces max concurrent image generations
- Lazy import avoids rate_limits module dependency in image_utils
- Two-stage: LLM generates prompt, then Imagen generates image

### Step 6: Centralized Resource Lifecycle

```python
# core/task_queue/lifecycle.py

"""Supervisor lifecycle helpers."""

import logging
from core.llm_broker import get_broker, is_broker_enabled
from core.utils.async_http_client import cleanup_all_clients

logger = logging.getLogger(__name__)


async def cleanup_supervisor_resources() -> None:
    """Clean up shared resources owned by the supervisor (broker, HTTP clients).

    Called from the finally block of both parallel.py and queue_loop.py.
    """
    if is_broker_enabled():
        try:
            await get_broker().stop()
        except Exception:
            logger.exception("Error stopping broker")

    await cleanup_all_clients()
```

**Key Details:**
- Shared by both `parallel.py` and `queue_loop.py` finally blocks
- Broker stop moved OUT of `workflow_executor.py` to avoid killing singleton
- HTTP client cleanup via registry pattern
- Exception handling prevents cleanup failures from masking other errors

### Step 7: Orphaned Task Recovery

```python
# core/task_queue/parallel.py (finally block)

    finally:
        coordinator.remove_signal_handlers()
        await cleanup_supervisor_resources()

        # Reset orphaned IN_PROGRESS tasks back to PENDING so they are
        # retried on the next invocation instead of staying stuck.
        try:
            def _reset_orphaned():
                with queue_manager.persistence.lock():
                    queue = queue_manager.persistence.read_queue()
                    reset_count = 0
                    for task in queue["topics"]:
                        if (
                            task["id"] in selected_ids
                            and task["status"] == "in_progress"
                        ):
                            task["status"] = "pending"
                            task.pop("started_at", None)
                            reset_count += 1
                    if reset_count:
                        queue_manager.persistence.write_queue(queue)
                        logger.info(
                            f"Reset {reset_count} orphaned task(s) back to PENDING"
                        )

            await asyncio.to_thread(_reset_orphaned)
        except Exception:
            logger.exception("Error resetting orphaned tasks")
```

**Key Details:**
- Runs in finally block (even if supervisor crashes or is cancelled)
- Only resets tasks that were selected by this supervisor run
- Uses `asyncio.to_thread()` for file lock safety
- Logs reset count for debugging
- Swallows exceptions to avoid masking other errors

### Step 8: CLI Entry with Auto-Scaling

```python
# core/task_queue/commands/parallel_command.py

def cmd_parallel(args):
    """Run multiple tasks concurrently."""
    if args.count < 1:
        print("Error: --count must be >= 1")
        sys.exit(1)
    if args.stagger < 0:
        print("Error: --stagger must be >= 0")
        sys.exit(1)

    # Auto-scale broker concurrency for parallel execution if not already configured
    if "THALA_LLM_BROKER_MAX_CONCURRENT_SYNC" not in os.environ:
        recommended = args.count * 3
        os.environ["THALA_LLM_BROKER_MAX_CONCURRENT_SYNC"] = str(recommended)
        print(f"Auto-scaling broker concurrency to {recommended} (count * 3)")

    print(f"Starting parallel execution: {args.count} tasks, {args.stagger}min stagger")
    results = asyncio.run(
        run_parallel_tasks(
            count=args.count,
            stagger_minutes=args.stagger,
        )
    )

    # Summary
    succeeded = sum(1 for r in results if not isinstance(r, BaseException))
    failed = len(results) - succeeded
    print(f"\nParallel execution complete: {succeeded} succeeded, {failed} failed")
```

**Key Details:**
- Auto-scales broker concurrency to `count * 3` (heuristic for 3 LLM calls/workflow)
- Only sets if not already configured (respects explicit override)
- Validation prevents invalid arguments
- Result summary counts exceptions vs successful dicts

## Complete Example

```python
# Example: Running 5 concurrent lit reviews with 3min stagger

# CLI usage
$ thala-queue parallel --count 5 --stagger 3.0

# What happens:
# 1. Auto-scales broker: THALA_LLM_BROKER_MAX_CONCURRENT_SYNC=15
# 2. Atomically selects 5 PENDING tasks, sets to IN_PROGRESS
# 3. Starts task 1 immediately
# 4. Starts task 2 after 3min, task 3 after 6min, etc.
# 5. Each workflow:
#    - Uses shared broker singleton (15 concurrent LLM calls max)
#    - Respects Imagen semaphore (10 concurrent max)
#    - Respects OpenAlex semaphore (20 concurrent max)
# 6. On completion/crash:
#    - Stops broker once (not per-workflow)
#    - Cleans up HTTP clients
#    - Resets orphaned IN_PROGRESS tasks to PENDING

# Result: 5 workflows complete in ~30min vs 100min sequential
```

### Configuring Rate Limits

```bash
# Override default semaphore limits
export THALA_IMAGEN_CONCURRENCY=5       # Default: 10
export THALA_OPENALEX_CONCURRENCY=30    # Default: 20
export THALA_LLM_BROKER_MAX_CONCURRENT_SYNC=20  # Default: 5 (auto-scales in parallel mode)

# Run with custom config
thala-queue parallel --count 3 --stagger 5.0
```

## Consequences

### Benefits

- **5x throughput increase**: 5 concurrent workflows vs sequential (15-30min each)
- **Staggered starts**: Prevent API thundering herd, spread budget usage
- **Per-API rate limiting**: Independent semaphores per API (Imagen, OpenAlex, etc.)
- **Orphaned task recovery**: Crashes don't leave tasks stuck IN_PROGRESS
- **Centralized lifecycle**: Broker singleton properly managed, not killed mid-flight
- **Lazy semaphore factories**: Work across `asyncio.run()` boundaries (tests)
- **File lock safety**: `asyncio.to_thread()` for all sync I/O operations
- **Budget bypass support**: Zero-cost workflows can run via registry pattern

### Trade-offs

- **Higher peak memory**: 5 concurrent workflow states vs 1 sequential
- **Complex error handling**: `return_exceptions=True` means caller checks types
- **Stagger latency**: Later tasks wait stagger_minutes * index before starting
- **Global semaphore limits**: One slow API call blocks others on same semaphore
- **Broker auto-scaling heuristic**: `count * 3` may over/under-provision
- **Orphaned task reset scope**: Only resets tasks selected by this supervisor run

### Alternatives

- **Celery distributed queue**: More complex, requires broker infrastructure
- **Ray parallel execution**: Heavyweight, requires cluster setup
- **Thread pool**: Limited by GIL, harder to debug async workflows
- **Process pool**: High overhead, no shared state (broker, HTTP clients)

### Gotchas

1. **Module-level semaphores go stale**: MUST use lazy factory pattern, not module-level init
2. **fcntl.flock() blocks event loop**: MUST use `asyncio.to_thread()` for file lock ops
3. **Broker stop was in workflow_executor**: Had to be removed and centralized to lifecycle.py
4. **Auto-scaling is heuristic**: May need explicit override for workflows with different LLM usage
5. **Semaphore limits are per-API, not per-workflow**: One slow workflow can block others
6. **return_exceptions=True changes semantics**: Caller must check if result is BaseException

## Testing Strategy

### Unit Tests for Parallel Supervisor

```python
# tests/unit/core/task_queue/test_parallel.py (290 lines)

# Task selection ordering
def test_select_tasks_priority_fifo_ordering():
    """Tasks are selected by priority desc, then created_at asc."""

# Empty queue handling
def test_run_parallel_tasks_empty_queue():
    """Returns empty list when no pending tasks."""

# Budget skip
async def test_run_parallel_tasks_budget_skip():
    """Task is skipped (not failed) when budget limit reached."""

# Shutdown during stagger
async def test_run_parallel_tasks_shutdown_during_stagger():
    """Later tasks are cancelled when shutdown during stagger delay."""

# Workflow exception handling
async def test_run_parallel_tasks_one_workflow_exception():
    """Exception in one workflow doesn't cancel others."""

# Orphaned task recovery
async def test_run_parallel_tasks_resets_orphaned_in_progress_tasks():
    """IN_PROGRESS tasks are reset to PENDING in finally block."""
```

### Unit Tests for Rate Limits

```python
# tests/unit/core/task_queue/test_rate_limits.py (107 lines)

# Lazy initialization
def test_imagen_semaphore_lazy_init():
    """Semaphore is created on first access."""

# Environment variable override
def test_imagen_semaphore_env_override():
    """THALA_IMAGEN_CONCURRENCY overrides default limit."""

# Independent semaphores
def test_semaphores_are_independent():
    """Imagen and OpenAlex semaphores don't share state."""

# Concurrent access
async def test_imagen_semaphore_limits_concurrency():
    """Semaphore enforces max concurrent calls."""
```

### Integration Testing

```bash
# Manual integration test
$ export THALA_IMAGEN_CONCURRENCY=2
$ export THALA_OPENALEX_CONCURRENCY=5
$ thala-queue parallel --count 3 --stagger 1.0

# Verify:
# - 3 tasks start with 1min stagger
# - Max 2 concurrent Imagen calls
# - Max 5 concurrent OpenAlex calls
# - Broker stops once after all complete
# - No orphaned IN_PROGRESS tasks
```

## Related Patterns

- [Multi-Workflow Task Queue](./multi-workflow-task-queue.md) - Registry-based polymorphic workflow dispatch
- [Task Queue Budget Tracking](./task-queue-budget-tracking.md) - Budget and checkpoint foundation
- [Incremental Checkpointing](./incremental-checkpointing-iterative-workflows.md) - Phase resumption patterns

## Known Uses in Thala

- `core/task_queue/parallel.py`: Parallel supervisor (160 lines)
- `core/task_queue/rate_limits.py`: Semaphore factories (31 lines)
- `core/task_queue/lifecycle.py`: Resource cleanup (23 lines)
- `core/task_queue/commands/parallel_command.py`: CLI entry (37 lines)
- `workflows/shared/image_utils.py`: Imagen semaphore consumer
- `langchain_tools/openalex/client.py`: OpenAlex semaphore consumer
- `core/task_queue/queue_loop.py`: Also uses lifecycle.py for cleanup

## References

- Commit: 1bc9aeb, 95a2041 (feat/parallel-lit-review-supervisor branch)
- [asyncio.gather()](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
- [asyncio.Semaphore](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore)
- [asyncio.to_thread()](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)
