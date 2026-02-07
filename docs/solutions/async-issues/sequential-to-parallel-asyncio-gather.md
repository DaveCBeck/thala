---
type: solution
title: Sequential to Parallel Async Processing with Semaphore
category: async-issues
created: 2026-02-06
severity: p1
module: workflows.shared.llm_utils
symptoms:
  - "Batch LLM calls taking O(n) time instead of O(1)"
  - "Sequential await in for-loop defeating async benefits"
  - "Poor performance on batch inputs"
  - "30+ second delays for 10 prompts that should run concurrently"
root_cause: sequential_await_defeats_async_benefits
resolution_type: code_fix
verified_fix: true
tags: [async, performance, asyncio-gather, semaphore, rate-limiting, batch-processing]
---

# Sequential to Parallel Async Processing with Semaphore

## Problem

The `_invoke_direct()` function processed multiple user prompts sequentially using a for-loop with `await`, defeating the entire purpose of async execution:

```python
# BAD: Sequential processing - O(n) latency
results = []
for user_prompt in user_prompts:
    messages = build_messages(system, user_prompt)
    response = await llm.ainvoke(messages)  # BLOCKS until complete
    results.append(response)
    # Next request starts ONLY after previous completes
```

**Observed performance:**
- 10 prompts: ~30 seconds (3s each sequentially)
- 100 prompts: ~5 minutes (3s each sequentially)

**Why this is wrong:**
- Each `await` blocks until the LLM response completes
- The next request only starts after the previous one finishes
- With 10 prompts taking 3s each, total time is 30s instead of 3s
- Async infrastructure provides zero value

## Root Cause

**Sequential await in for-loop defeats async concurrency.**

When you write:
```python
for item in items:
    result = await async_function(item)
```

You're forcing sequential execution. The `await` blocks until `async_function()` completes before moving to the next iteration.

## Solution

**Use `asyncio.gather()` with semaphore for concurrent execution with rate limiting.**

```python
# GOOD: Parallel processing with rate limiting - O(1) latency
semaphore = asyncio.Semaphore(config.max_concurrent)

async def invoke_one(user_prompt: str) -> AIMessage:
    async with semaphore:
        # Build messages with caching for Anthropic
        if not is_deepseek_tier(tier) and config.cache:
            messages = create_cached_messages(
                system_content=system,
                user_content=user_prompt,
                cache_system=True,
                cache_ttl=config.cache_ttl,
            )
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        return await llm.ainvoke(messages)

return list(await asyncio.gather(*[invoke_one(p) for p in user_prompts]))
```

### Key Implementation Points

**1. asyncio.gather() preserves result order**
```python
# Even though tasks complete in random order, gather() returns results
# in the same order as the input coroutines
results = await asyncio.gather(coro1, coro2, coro3)
# results[0] is from coro1, results[1] from coro2, results[2] from coro3
```

**2. Semaphore prevents rate limiting (429 errors)**
```python
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

async def invoke_one(prompt):
    async with semaphore:  # Blocks if 10 requests already running
        return await llm.ainvoke(...)
```

**3. Default max_concurrent = 10 is sensible**
- Anthropic rate limits: 50 req/min for Sonnet
- 10 concurrent = ~6 req/sec = 360 req/min (safe margin)
- Too high: risk 429 rate limit errors
- Too low: underutilize async benefits

## Performance Improvement

| Batch Size | Sequential | Parallel (10 concurrent) | Speedup |
|------------|------------|--------------------------|---------|
| 10 prompts | ~30s | ~3s | **10x** |
| 50 prompts | ~2.5 min | ~15s | **10x** |
| 100 prompts | ~5 min | ~30s | **10x** |

**Assumptions:**
- Each LLM call takes ~3 seconds
- Rate limits not exceeded
- Network latency not a bottleneck

## Implementation

### Before (Sequential)

```python
async def _invoke_direct(
    tier: ModelTier,
    system: str,
    user_prompts: list[str],
    config: InvokeConfig,
) -> list[AIMessage]:
    llm = get_llm(tier=tier, thinking_budget=config.thinking_budget)

    results = []
    for user_prompt in user_prompts:  # Sequential!
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        response = await llm.ainvoke(messages)  # BLOCKS
        results.append(response)

    return results
```

### After (Parallel with Semaphore)

```python
async def _invoke_direct(
    tier: ModelTier,
    system: str,
    user_prompts: list[str],
    config: InvokeConfig,
) -> list[AIMessage]:
    llm = get_llm(
        tier=tier,
        thinking_budget=config.thinking_budget,
        max_tokens=config.max_tokens,
    )

    # Rate limiting semaphore
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def invoke_one(user_prompt: str) -> AIMessage:
        async with semaphore:
            # Build messages with caching for Anthropic
            if not is_deepseek_tier(tier) and config.cache:
                messages = create_cached_messages(
                    system_content=system,
                    user_content=user_prompt,
                    cache_system=True,
                    cache_ttl=config.cache_ttl,
                )
            else:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ]
            return await llm.ainvoke(messages)

    return list(await asyncio.gather(*[invoke_one(p) for p in user_prompts]))
```

## Configuration

```python
from workflows.shared.llm_utils import InvokeConfig

# Default (10 concurrent)
config = InvokeConfig()

# Aggressive (20 concurrent) - risk rate limits
config = InvokeConfig(max_concurrent=20)

# Conservative (5 concurrent) - safer for high-volume
config = InvokeConfig(max_concurrent=5)
```

## Files Modified

**Core implementation:**
- `workflows/shared/llm_utils/invoke.py` - Changed `_invoke_direct()` to use `asyncio.gather()`

**Configuration:**
- `workflows/shared/llm_utils/config.py` - Added `max_concurrent` field to `InvokeConfig`

## Error Handling

```python
# asyncio.gather() propagates exceptions by default
try:
    results = await asyncio.gather(*tasks)
except Exception as e:
    # First exception from any task is raised
    logger.error(f"Batch processing failed: {e}")
    raise

# Alternative: return_exceptions=True to handle individually
results = await asyncio.gather(*tasks, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Task {i} failed: {result}")
```

## Common Async Antipatterns

### Antipattern 1: Sequential await in for-loop
```python
# BAD - O(n) latency
for item in items:
    result = await process(item)
```

### Antipattern 2: Creating tasks without gather
```python
# BAD - tasks start but results are lost
tasks = [asyncio.create_task(process(item)) for item in items]
# Need to await them!
```

### Antipattern 3: No rate limiting
```python
# BAD - may trigger rate limits
results = await asyncio.gather(*[process(item) for item in items])
# Should use semaphore for external API calls
```

## When to Use asyncio.gather()

**Use asyncio.gather() when:**
- Processing multiple independent async operations
- Order of results matters (gather preserves order)
- Want simple error handling (first exception propagates)
- External API calls need rate limiting (combine with semaphore)

**Don't use asyncio.gather() when:**
- Tasks have dependencies (use sequential await)
- Need fine-grained control over task lifecycle (use TaskGroup or manual tasks)
- Streaming results as they arrive (use async iterators)
- Tasks should continue even if one fails (use `return_exceptions=True`)

## Related Solutions

- [HTTP Client Cleanup Registry](./http-client-cleanup-registry.md) - Resource management for async HTTP clients
- [Streaming Async Results Pipeline](./streaming-async-results-pipeline.md) - Producer-consumer patterns for streaming

## Related Patterns

- [Parallel Workflow Supervisor](../../patterns/data-pipeline/parallel-workflow-supervisor.md) - Applies gather + semaphore at workflow level with staggered starts and shared resource lifecycle
- **Concurrent Scraping with TTL Cache** (`docs/patterns/async-python/concurrent-scraping-with-ttl-cache.md`) - Semaphore + caching
- **Streaming Producer-Consumer Pipeline** (`docs/patterns/async-python/streaming-producer-consumer-pipeline.md`) - AsyncIterator patterns

## References

- [asyncio.gather() documentation](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
- [asyncio.Semaphore documentation](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore)
- [Anthropic API Rate Limits](https://docs.anthropic.com/en/api/rate-limits)

## Prevention

When writing async batch processing code:

1. **Never use sequential await in for-loop for independent operations**
   ```python
   # BAD
   for item in items:
       await process(item)

   # GOOD
   await asyncio.gather(*[process(item) for item in items])
   ```

2. **Always add rate limiting for external API calls**
   ```python
   semaphore = asyncio.Semaphore(10)

   async def process_with_limit(item):
       async with semaphore:
           return await api_call(item)
   ```

3. **Verify actual concurrency in tests**
   ```python
   # Add timing assertions to catch sequential execution
   start = time.time()
   results = await process_batch(items)
   elapsed = time.time() - start
   assert elapsed < len(items) * expected_per_item  # Should be concurrent!
   ```

4. **Use logging to verify parallel execution**
   ```python
   logger.info(f"Starting batch of {len(items)} items")
   results = await asyncio.gather(*tasks)
   logger.info(f"Completed in {elapsed:.2f}s (expected ~{expected:.2f}s)")
   ```
