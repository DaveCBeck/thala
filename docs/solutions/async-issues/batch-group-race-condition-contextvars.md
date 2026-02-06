---
module: core/llm_broker
date: 2026-02-06
problem_type: async_issue
component: broker
symptoms:
  - "Requests assigned to wrong batch group"
  - "Parallel LangGraph nodes interfering with each other's batches"
  - "Batch results not matching expected requests"
  - "Race conditions in concurrent async contexts"
root_cause: shared_state
resolution_type: code_fix
severity: p1
tags: [async, contextvars, race-condition, langgraph, broker, concurrency, batch-processing]
---

# Batch Group Race Condition with ContextVars

## Problem

The LLM broker used a single instance variable `self._current_group` to track the active batch group. When parallel LangGraph nodes or concurrent async tasks called broker methods simultaneously, they interfered with each other's batch groups, causing requests to be assigned to the wrong batch.

### Symptoms

1. **Wrong batch assignment**: Requests from one async context added to another context's batch group.

2. **Orphaned requests**: Requests added during context setup but lost when another context overwrote the group.

3. **LangGraph node interference**: Parallel nodes in LangGraph workflows (using `Send()` for fan-out) corrupted each other's batch tracking.

4. **Non-deterministic failures**: Race condition timing made failures intermittent and hard to reproduce.

### Race Condition Scenario

```python
# Two LangGraph nodes executing concurrently
async def node_a():
    async with broker.batch_group():  # Step 1: Sets _current_group = groupA
        # Step 3: Tries to add request, but _current_group = groupB now!
        await broker.request("prompt A")  # Goes to wrong group!
        # Step 4: Exits, sets _current_group = None
        # Step 5: groupB's requests are now orphaned

async def node_b():
    async with broker.batch_group():  # Step 2: Sets _current_group = groupB (overwrites!)
        await broker.request("prompt B")
```

**Timeline:**
1. Node A enters `batch_group()`, sets `_current_group = groupA`
2. Node B enters `batch_group()`, sets `_current_group = groupB` (overwrites!)
3. Node A adds request - it goes to groupB (wrong group!)
4. Node A exits, sets `_current_group = None`
5. Node B's requests are orphaned (current group is None)

### Example Failure Log

```
INFO: Node A: Creating batch group
INFO: Node B: Creating batch group
DEBUG: _current_group set to group_b (overwrote group_a)
INFO: Node A: Adding request req_a to batch
WARNING: Request req_a added to group_b (expected group_a)
INFO: Node A: Exiting batch group, flushing 0 requests  # Lost!
ERROR: Node B: Batch group incomplete, missing requests
```

## Root Cause

### Shared Instance Variable

The broker stored the current batch group as an instance variable:

```python
# BEFORE (problematic)
class LLMBroker:
    def __init__(self):
        self._current_group: BatchGroup | None = None

    @asynccontextmanager
    async def batch_group(self):
        group = BatchGroup(broker=self)
        self._current_group = group  # Shared state!
        try:
            yield group
        finally:
            self._current_group = None
            await self._flush_batch_group(group)

    async def _queue_for_batch(self, request):
        # Reads shared state
        if self._current_group:
            self._current_group.add_request(request.request_id)
```

### Why This Fails in Async

Python's async runtime allows task switching at any `await` point. Multiple async contexts can be interleaved even in single-threaded execution:

1. **No thread-local storage**: Unlike threading, async tasks share the same thread, so `threading.local()` doesn't help.
2. **Task interleaving**: When `node_a` awaits, `node_b` can run and modify shared state.
3. **Instance variable shared**: All concurrent calls to the same broker instance see the same `self._current_group`.

This is especially problematic in LangGraph workflows that use `Send()` for parallel node execution, where multiple nodes run concurrently and all use the same global broker instance.

## Solution

### Use ContextVars for Isolation

Python's `contextvars` module provides async-safe context isolation. Each async context gets its own copy of the variable, preventing interference:

```python
# AFTER (fixed)
from contextvars import ContextVar

# Module-level context variable
_current_batch_group: ContextVar["BatchGroup | None"] = ContextVar(
    "llm_broker_batch_group", default=None
)

class LLMBroker:
    @asynccontextmanager
    async def batch_group(self, mode: UserMode | None = None):
        """Context manager for grouping requests into a batch.

        Uses contextvars for isolation, ensuring parallel LangGraph nodes
        or concurrent async contexts don't interfere with each other's
        batch groups.
        """
        group = BatchGroup(broker=self, mode=mode or self._mode)
        token = _current_batch_group.set(group)  # Set with token

        try:
            yield group
        finally:
            _current_batch_group.reset(token)  # Reset to previous value

            # Flush queued requests from this group
            if group.request_ids:
                await self._flush_batch_group(group)

    async def _queue_for_batch(self, request: LLMRequest):
        # Read from context variable (isolated per async context)
        current_group = _current_batch_group.get()
        if current_group:
            current_group.add_request(request.request_id)
```

### Key Changes

1. **Module-level ContextVar**: Declared at module scope, not instance variable.
2. **Token-based reset**: `set()` returns a token, `reset(token)` restores previous value.
3. **Isolated reads**: Each async context sees its own value via `get()`.
4. **Default value**: `default=None` provides fallback when no context is set.

## Why ContextVars Work

### Async Context Isolation

ContextVars maintain separate values for each async task/context:

```python
# Example: Two concurrent contexts
async def context_1():
    token = _current_batch_group.set(group1)
    # This context sees group1
    assert _current_batch_group.get() == group1
    await asyncio.sleep(0)  # Task switch point
    # Still sees group1, even if context_2 ran
    assert _current_batch_group.get() == group1
    _current_batch_group.reset(token)

async def context_2():
    token = _current_batch_group.set(group2)
    # This context sees group2 (isolated!)
    assert _current_batch_group.get() == group2
    _current_batch_group.reset(token)

await asyncio.gather(context_1(), context_2())
```

### How It Works

1. **Context creation**: `set()` creates a new context value and returns a reset token.
2. **Propagation**: Child tasks inherit the current context value (copy-on-write).
3. **Isolation**: Modifications in one context don't affect others.
4. **Cleanup**: `reset(token)` restores the previous value from before `set()`.

### Why Not Alternatives?

| Solution | Problem |
|----------|---------|
| `threading.local()` | Doesn't work for async - all tasks share same thread |
| Global dict + task ID | Requires manual cleanup, prone to leaks |
| Locks/semaphores | Serializes concurrent contexts, defeats parallelism |
| Task-local storage | No built-in mechanism in asyncio |

ContextVars are the **standard Python solution** for async-safe context isolation (PEP 567).

## Prevention

### When to Use ContextVars

Use ContextVars for state that should be isolated per async context:

1. **Request scoping**: Current user, request ID, batch group
2. **Tracing context**: Span IDs, trace metadata
3. **Configuration overrides**: Per-request settings
4. **Resource tracking**: Context-specific connections, transactions

**Rule of thumb**: If multiple async operations need different values of the same logical state, use ContextVar.

### Pattern Template

```python
from contextvars import ContextVar
from contextlib import asynccontextmanager

# Module-level declaration
_my_context: ContextVar[MyType | None] = ContextVar(
    "descriptive_name",
    default=None
)

@asynccontextmanager
async def my_context_manager(value: MyType):
    """Context manager with async isolation."""
    token = _my_context.set(value)
    try:
        yield value
    finally:
        _my_context.reset(token)
        # Optional cleanup
        await cleanup(value)

# Usage
async def my_function():
    current = _my_context.get()  # Read current context
    if current:
        current.do_something()
```

### Testing for Race Conditions

Test concurrent usage explicitly:

```python
async def test_parallel_batch_groups():
    """Test that parallel batch groups are isolated."""
    broker = LLMBroker()
    await broker.start()

    results = []

    async def create_group(group_id: int):
        async with broker.batch_group() as group:
            await broker.request(f"prompt {group_id}")
            results.append((group_id, len(group.request_ids)))

    # Run parallel groups
    await asyncio.gather(
        create_group(1),
        create_group(2),
        create_group(3),
    )

    # Each group should have exactly 1 request (no interference)
    assert results == [(1, 1), (2, 1), (3, 1)]
```

### Warning Signs

Watch for these patterns that indicate potential race conditions:

1. **Instance variables for context state**: Use ContextVar instead
2. **Shared mutable state across async calls**: Isolate or protect with locks
3. **Assumptions about task ordering**: Async is non-deterministic
4. **Global singletons with request state**: Use ContextVar for per-request data

## Files Modified

- `core/llm_broker/broker.py`: Added `_current_batch_group` ContextVar, updated `batch_group()` and `_queue_for_batch()`

## Related Patterns

- [Central LLM Broker Pattern](../../patterns/llm-broker/central-broker-pattern.md) - Broker architecture
- [Async Context Management](../../patterns/async/context-management.md) - ContextVar best practices

## References

- [PEP 567 - Context Variables](https://peps.python.org/pep-0567/) - Official specification
- [Python contextvars documentation](https://docs.python.org/3/library/contextvars.html)
- [LangGraph Send() for parallel nodes](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.MessageGraph.send)
