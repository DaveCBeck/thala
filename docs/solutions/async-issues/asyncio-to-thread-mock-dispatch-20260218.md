---
module: core/task_queue
date: 2026-02-18
problem_type: async_issue
component: parallel
symptoms:
  - "Tests pass with one asyncio.to_thread call but break when a second is added"
  - "Mock return_value shared across all to_thread calls"
  - "TypeError or unexpected type from asyncio.to_thread mock"
root_cause: blanket_mock
resolution_type: test_fix
severity: p2
tags: [async, testing, mocking, asyncio, to-thread, side-effect, parallel]
---

# asyncio.to_thread Mock Dispatch for Multiple Wrapped Calls

## Problem

When production code uses `asyncio.to_thread()` to offload multiple different sync functions, a blanket mock with `return_value` makes all calls return the same value — breaking tests that depend on each function returning its own type.

### Symptoms

```python
# Production code — TWO different to_thread calls:
tasks = await asyncio.to_thread(_select_tasks, queue_manager, count, ids)
should_proceed, reason = await asyncio.to_thread(budget_tracker.should_proceed)
```

```python
# Test — blanket mock returns the SAME value for both:
@patch("core.task_queue.parallel.asyncio.to_thread", return_value=[task])
```

The budget check receives `[task]` (a list) instead of `(True, "")` (a tuple), causing `TypeError` or wrong control flow.

## Root Cause

`Mock(return_value=X)` returns `X` for every call regardless of arguments. When `asyncio.to_thread` wraps multiple callables with different return types, a single `return_value` cannot satisfy all of them.

## Solution

Use `side_effect` with a dispatch function that inspects which callable was passed to `to_thread`:

```python
def _fake_to_thread(select_result):
    """Build a side_effect for asyncio.to_thread that dispatches by callable."""

    async def _side_effect(fn, *args, **kwargs):
        if fn is _select_tasks:
            return select_result
        # All other callables (budget_tracker.should_proceed, etc.)
        # are called through so their own mock return_values work.
        return fn(*args, **kwargs)

    return _side_effect
```

Usage in tests:

```python
@patch("core.task_queue.parallel.asyncio.to_thread",
       side_effect=_fake_to_thread([task]))
```

### Why This Works

- `fn is _select_tasks` — identity check intercepts the specific function to stub
- `fn(*args, **kwargs)` — call-through for everything else preserves each collaborator's own mock behaviour (e.g. `budget.should_proceed.return_value = (True, "")`)
- Adding a new `to_thread` call in production doesn't break existing tests — the call-through handles unknown callables automatically

## Prevention

When wrapping sync I/O in `asyncio.to_thread`:

1. **Never blanket-mock `asyncio.to_thread` with `return_value`** if the function under test calls it more than once
2. **Use dispatch-by-callable `side_effect`** from the start — it handles single calls fine and scales to multiple
3. **Identity-check (`fn is X`)** is more robust than string matching on function names

## Related

- [Batch Group Race Condition with ContextVars](batch-group-race-condition-contextvars.md) — async context isolation
- [Three-Layer Rate Limiting](../../patterns/async-python/three-layer-rate-limiting.md) — uses `asyncio.to_thread` for file-backed sync ops
- [Task Queue Interruption Recovery](../workflow-reliability/task-queue-interruption-recovery.md) — `asyncio.to_thread` in shutdown paths
