---
name: three-layer-rate-limiting
title: "Three-Layer Rate Limiting for External API Calls"
date: 2026-02-15
category: async-python
applicability:
  - "External APIs with both daily quotas and per-minute rate limits"
  - "Concurrent async tasks sharing a single API budget"
  - "APIs where overshoot has hard consequences (billing, bans, not just 429s)"
components: [daily_tracker, rpm_limiter, concurrency_semaphore]
complexity: moderate
verified_in_production: false
related_solutions: []
related_patterns:
  - "docs/patterns/data-pipeline/task-queue-budget-tracking.md"
  - "docs/patterns/data-pipeline/parallel-workflow-supervisor.md"
tags: [rate-limiting, asyncio, semaphore, token-bucket, flock, imagen, api]
---

# Three-Layer Rate Limiting for External API Calls

## Intent

Protect an expensive external API (Google Imagen) from quota overruns by composing three rate-limiting layers checked cheapest-first, so rejected requests waste minimal work.

## Motivation

A single rate limiter cannot protect an API with multiple constraint dimensions. Concurrent async tasks can blow through a daily quota in minutes, exceed per-minute caps triggering 429s, or open too many sockets at once. Each failure mode needs a different control mechanism at a different time scale.

**Single-layer failure modes:**
```
Semaphore only   -> burns through daily quota in minutes
RPM limiter only -> 50 tasks wake up at midnight, all pass RPM, exhaust daily budget
Daily check only -> 5 requests fire simultaneously, all pass daily check before any decrement
```

## Implementation

Three layers, checked cheapest-first:

```
Request → [Daily budget] → [RPM token bucket] → [Concurrency semaphore] → API call
              ↓ fail            ↓ full                ↓ blocked
           DEFERRED          async sleep            async wait
```

### Layer 1: Daily Budget (file-backed, cross-process safe)

Atomic check-and-decrement under `flock` prevents TOCTOU races between OS processes.

```python
def _try_acquire_sync(self, count: int = 1) -> bool:
    with self._file_lock():          # fcntl.flock(LOCK_EX)
        data = self._read_state()    # {"date": "2026-02-15", "count": 7}
        if data["date"] != _today_str():
            data = {"date": _today_str(), "count": 0}
        if data["count"] + count > self._limit:
            return False             # budget exhausted -- caller defers
        data["count"] += count
        write_json_atomic(self._state_file, data)
        return True
```

### Layer 2: RPM Token Bucket (in-memory, async)

Refills tokens proportionally to elapsed time. Sleeps the caller when empty rather than rejecting.

```python
async def acquire(self, cost: int = 1) -> None:
    while True:
        async with self._lock:
            elapsed = now - self._last_refill
            self._tokens = min(float(self._rpm),
                               self._tokens + elapsed * (self._rpm / 60.0))
            if self._tokens >= cost:
                self._tokens -= cost
                return
        await asyncio.sleep(60.0 / self._rpm)   # yield, then retry
```

### Layer 3: Concurrency Semaphore

Standard `asyncio.Semaphore` caps in-flight API calls to prevent socket/memory exhaustion.

### Composition

```python
if not await daily_tracker.try_acquire(sample_count):  # Layer 1: cheapest check
    return None, prompt                                 # DEFERRED -- no API call

await rpm_limiter.acquire(sample_count)                 # Layer 2: may sleep

async with get_imagen_semaphore():                      # Layer 3: cap concurrency
    response = await asyncio.wait_for(
        client.aio.models.generate_images(...),
        timeout=GOOGLE_API_TIMEOUT,
    )
```

Both `THALA_IMAGEN_DAILY_LIMIT` and `THALA_IMAGEN_RPM_LIMIT` are expressed in **images** (not requests). A single API call with `sample_count=4` consumes 4 from each budget.

## Consequences

**Benefits:**
- Daily budget never exceeded (file-locked atomic counter)
- RPM stays under API limit (token bucket with proportional refill)
- No socket/memory exhaustion (semaphore bounds concurrency)
- Cheapest-first ordering avoids wasting budget slots or RPM tokens on requests that would be blocked downstream

**Trade-offs:**
- File-based daily tracker requires `asyncio.to_thread` wrapper for async contexts
- Three layers add complexity; only warranted for APIs with genuine multi-dimensional limits

## Gotchas

1. **Lazy factories, not module globals.** `asyncio.Semaphore` created at import time goes stale across `asyncio.run()` boundaries (common in tests and daemon restarts). All primitives use `get_*()` factories with a `reset_rate_limiters()` teardown hook.

2. **`asyncio.to_thread` for `flock`.** The daily tracker wraps its synchronous file-lock path in `asyncio.to_thread` so `flock` blocking never stalls the event loop.

3. **Order matters.** Daily check is pure read + file I/O (cheap); RPM may sleep but costs no API call; the semaphore gates the actual network request. Reversing the order wastes budget on requests that would have been blocked anyway.

4. **Don't double-consume budget.** If both a workflow outer loop and the inner API function call `try_acquire()`, each request consumes 2 daily slots. The outer loop should use a non-consuming `remaining()` check; let the API function be the single source of truth.

## Known Uses

- `core/task_queue/rate_limits.py` -- `ImagenDailyTracker`, `ImagenRPMLimiter`, `get_imagen_semaphore()`
- `workflows/shared/image_utils.py:generate_article_header()` -- composes all three layers
- `core/task_queue/workflows/illustrate_and_publish.py` -- outer loop uses `remaining()` for fast-fail

## Related

- [Task Queue Budget Tracking](../data-pipeline/task-queue-budget-tracking.md) -- broader budget tracking pattern
- [Parallel Workflow Supervisor](../data-pipeline/parallel-workflow-supervisor.md) -- semaphore-based rate limiting in parallel execution
