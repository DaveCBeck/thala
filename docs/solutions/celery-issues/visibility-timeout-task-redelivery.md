---
module: services/marker
date: 2026-01-18
problem_type: configuration_error
component: celery_broker
symptoms:
  - "Long-running tasks (>1 hour) being processed by multiple workers simultaneously"
  - "Duplicate document conversions appearing in results"
  - "Tasks completing successfully but then being redelivered to another worker"
  - "GPU resource contention from concurrent processing of same task"
root_cause: broker_misconfiguration
resolution_type: config_fix
severity: high
tags: [celery, redis, visibility-timeout, task-redelivery, long-running-tasks, gpu, marker]
---

# Celery Visibility Timeout Causing Task Redelivery

## Problem

Long-running Celery tasks (>1 hour) are being processed by multiple workers simultaneously, causing duplicate work, resource contention, and corrupted results.

### Symptoms

```
# Worker 1 logs (started task at 10:00)
[2026-01-18 10:00:00] Received task: marker.process_pdf[abc123]
[2026-01-18 11:00:05] Processing PDF: 500 pages...

# Worker 2 logs (same task redelivered at 11:00)
[2026-01-18 11:00:05] Received task: marker.process_pdf[abc123]
[2026-01-18 11:00:06] Processing PDF: 500 pages...

# Both workers now process the same document concurrently
# Results: duplicate output, GPU contention, race conditions
```

Observable issues:
- Same PDF processed 2-3+ times simultaneously
- GPU memory exhaustion from concurrent identical tasks
- Corrupted output from race conditions
- Non-idempotent operations failing

### Impact

- **Wasted compute**: 4-hour GPU tasks duplicated = 8+ hours of GPU time per document
- **Resource exhaustion**: Multiple workers competing for same GPU memory
- **Data corruption**: Concurrent writes to same output files
- **Queue instability**: Cascading redeliveries creating infinite loops

## Root Cause

**Redis broker visibility_timeout was shorter than task execution time.**

### What is visibility_timeout?

When Celery receives a task from Redis, the message becomes "invisible" to other workers for `visibility_timeout` seconds. If the task isn't acknowledged within that time, Redis assumes the worker died and redelivers the task to another worker.

**Default Redis visibility_timeout: 3600 seconds (1 hour)**

### The Misconfiguration

```python
# Before: visibility_timeout effectively at default (1 hour)
celery.conf.update(
    task_time_limit=14400,  # 4 hours - but visibility_timeout was only 1 hour!
    broker_transport_options={
        "visibility_timeout": 18000,  # 5 hours - set inside update(), didn't take effect
    },
)
```

**Why this failed:**
1. `broker_transport_options` inside `conf.update()` wasn't being applied correctly
2. The effective visibility_timeout was the 1-hour default
3. Any task taking >1 hour got redelivered while still running

### The Relationship Problem

```
task_time_limit = 4 hours (14400s)
visibility_timeout = 1 hour (3600s default)

Timeline for a 3-hour task:
  [Hour 0] Worker A receives task
  [Hour 1] Redis redelivers to Worker B (visibility expired)
  [Hour 2] Redis redelivers to Worker C (visibility expired again)
  [Hour 3] Workers A, B, C all complete "successfully" → 3 duplicate results
```

## Solution

### 1. Set visibility_timeout Separately After App Creation

```python
# services/marker/app/tasks.py

celery = Celery(
    "marker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery.conf.update(
    task_soft_time_limit=10800,   # 3 hours soft limit
    task_time_limit=14400,         # 4 hours hard limit
    worker_prefetch_multiplier=1,  # One task at a time for GPU
    task_acks_late=True,           # Only ack after completion
    task_acks_on_failure_or_timeout=True,  # NEW: Ack failures to prevent requeue
    task_reject_on_worker_lost=True,
)

# CRITICAL: Set broker_transport_options separately after app creation
# This ensures it takes effect. Visibility timeout MUST exceed task_time_limit.
celery.conf.broker_transport_options = {
    "visibility_timeout": 28800,  # 8 hours (2x task_time_limit for safety margin)
}
```

### 2. Key Configuration Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| `visibility_timeout` | 28800s (8h) | 2x task_time_limit prevents premature redelivery |
| `task_acks_late` | True | Only acknowledge after task completes successfully |
| `task_acks_on_failure_or_timeout` | True | Ack failed/timed-out tasks to prevent infinite retry |
| `task_reject_on_worker_lost` | True | Don't requeue if worker dies mid-task |
| `worker_prefetch_multiplier` | 1 | One task per worker (critical for GPU isolation) |

### 3. The 2x Rule

**Best practice: visibility_timeout should be at least 2x task_time_limit**

```python
# Good: 8 hours >> 4 hours
visibility_timeout = 28800   # 8 hours
task_time_limit = 14400      # 4 hours

# Why 2x?
# - Provides safety margin for tasks near the time limit
# - Accounts for network latency, system slowdowns
# - Prevents cascading redeliveries during peak load
```

### 4. Why Set broker_transport_options Separately?

The `broker_transport_options` dict needs to be set **after** Celery app initialization:

```python
# ❌ WRONG: Inside conf.update() - may not apply correctly
celery.conf.update(
    broker_transport_options={"visibility_timeout": 28800},
)

# ✅ CORRECT: Separate assignment after initialization
celery.conf.broker_transport_options = {
    "visibility_timeout": 28800,
}
```

This ensures the broker connection receives the settings after it's initialized.

## Files Modified

- `services/marker/app/tasks.py` - Celery configuration with correct visibility_timeout

## Prevention

### 1. Always Verify visibility_timeout for Long-Running Tasks

```python
# Before deploying any task with time_limit > 1 hour, verify:
assert celery.conf.broker_transport_options.get("visibility_timeout", 3600) > task_time_limit * 2
```

### 2. Monitor for Duplicate Task Execution

```python
# Add idempotency check at task start
@celery.task(bind=True)
def process_pdf(self, file_path: str):
    # Check if another worker already processed this
    lock_key = f"processing:{file_path}"
    if redis_client.get(lock_key):
        logger.warning(f"Task {self.request.id} skipped - already being processed")
        return {"status": "skipped_duplicate"}

    redis_client.setex(lock_key, task_time_limit, self.request.id)
    try:
        # Process...
    finally:
        redis_client.delete(lock_key)
```

### 3. Log Visibility Timeout at Startup

```python
logger.info(
    f"Celery configured: task_time_limit={celery.conf.task_time_limit}, "
    f"visibility_timeout={celery.conf.broker_transport_options.get('visibility_timeout')}"
)
```

## Related Solutions

- [Large Document Processing](../api-integration-issues/large-document-processing.md) - Memory monitoring for long tasks
- [Workflow Reliability Retry Logic](../workflow-reliability/multi-signal-completeness-and-retry-logic.md) - Transient error handling

## Related Patterns

- [GPU-Accelerated Document Processing](../../patterns/data-pipeline/gpu-accelerated-document-processing.md) - Celery + GPU architecture
- [Phased Pipeline Architecture](../../patterns/data-pipeline/phased-pipeline-architecture-gpu-queue.md) - Queue management patterns

## References

- [Celery Redis Broker Documentation](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html)
- [Celery Issue #5935 - Long running jobs redelivering](https://github.com/celery/celery/issues/5935)
- [Configuring Celery for Reliable Delivery](https://www.francoisvoron.com/blog/configure-celery-for-reliable-delivery)
