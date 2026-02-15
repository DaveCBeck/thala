---
name: deferred-task-lifecycle
title: "DEFERRED Task Lifecycle for Budget-Aware Retry"
date: 2026-02-15
category: data-pipeline
applicability:
  - "Task queues where workflows can partially succeed before hitting resource limits"
  - "Budget-constrained APIs with daily/hourly quotas"
  - "Long-running per-item loops that should resume, not restart, after deferral"
components: [task_status, scheduler, workflow_executor, task_selector]
complexity: moderate
verified_in_production: false
related_solutions:
  - "docs/solutions/workflow-reliability/task-queue-interruption-recovery.md"
related_patterns:
  - "docs/patterns/data-pipeline/task-queue-budget-tracking.md"
  - "docs/patterns/data-pipeline/multi-workflow-task-queue.md"
  - "docs/patterns/async-python/three-layer-rate-limiting.md"
tags: [task-queue, deferred, retry, budget, scheduling, checkpoint, lifecycle]
---

# DEFERRED Task Lifecycle for Budget-Aware Retry

## Intent

Add a fifth task status (`DEFERRED`) to represent tasks that partially succeeded and need to resume later at a specific time, without losing progress or conflicting with existing PENDING/FAILED semantics.

## Motivation

The original `PENDING / IN_PROGRESS / COMPLETED / FAILED` lifecycle assumes a workflow either finishes or dies. It cannot represent a task that **partially succeeds** and needs to resume later -- for example, an illustration workflow that processes 2 of 5 articles before hitting a daily API budget limit.

| Option | Problem |
|--------|---------|
| Mark FAILED | Triggers alerts, loses "retry later" semantics |
| Mark PENDING | Re-runs from scratch, losing progress |
| Mark COMPLETED | Lies -- articles remain unillustrated |
| Keep IN_PROGRESS | Blocks the slot, orphan detection may kill it |

## Implementation

### Status Transition

```
PENDING -> IN_PROGRESS -> DEFERRED (next_run_after=now+3h)
                              |
                              v  (3 hours later, scheduler picks it up)
                          IN_PROGRESS -> COMPLETED
```

Per-article progress is stored in `task["items"]` so the task resumes where it left off.

### Workflow Returns "deferred"

```python
# illustrate_and_publish.py
if await daily_tracker.remaining() < 1:
    break  # exit per-article loop

next_run = (datetime.now(timezone.utc) + timedelta(hours=DEFER_HOURS)).isoformat()
return {"status": "deferred", "next_run_after": next_run, "items": items}
```

### Executor Transitions Task

```python
# workflow_executor.py
elif result_status == "deferred":
    queue_manager.update_task(task_id,
        status=TaskStatus.DEFERRED.value,
        next_run_after=next_run, started_at=None)
    checkpoint_mgr.complete_work(task_id)  # clear stale checkpoint
```

### Scheduler Filters by Time

All three scheduling paths (sequential, parallel, bypass) use the same check:

```python
# scheduler.py, parallel.py, task_selector.py
elif t["status"] == TaskStatus.DEFERRED.value:
    next_run = t.get("next_run_after")
    if not next_run or now >= datetime.fromisoformat(next_run):
        candidates.append(t)
```

### Resume Skips Completed Items

```python
# illustrate_and_publish.py -- per-article loop
for item in items:
    if item.get("draft_id"):
        continue  # already published in prior run
```

## Consequences

**Benefits:**
- Partial progress preserved across budget boundaries
- Scheduler respects `next_run_after` -- no tight retry loops
- Clean semantic distinction from FAILED (no false alerts)
- Works with both sequential and parallel executors

**Trade-offs:**
- Fifth status adds complexity to every scheduling path
- All three schedulers (sequential, parallel, bypass) must implement the `next_run_after` filter consistently

## Gotchas

1. **Clear the checkpoint on defer.** Leaving a stale checkpoint with a dead PID causes double-scheduling: `get_incomplete_work()` finds it AND the scheduler selects it via DEFERRED status.

2. **Reset `started_at=None` on defer.** Otherwise the task looks "in progress" to orphan-detection logic in the parallel supervisor's finally block.

3. **Progress lives in `task["items"]`, not checkpoint `phase_outputs`.** This avoids replaying the checkpoint-resume machinery; the per-article loop simply skips items where `item.get("draft_id")` is truthy.

4. **Malformed `next_run_after` is treated as immediately eligible** (defensive parsing with try/except) to avoid permanently stuck tasks.

5. **Don't forget bypass task selection.** `_find_bypass_task()` in `task_selector.py` must also filter by `next_run_after` -- this was missed initially and caused tight re-run loops in sequential mode.

## Known Uses

- `core/task_queue/schemas/enums.py` -- `TaskStatus.DEFERRED`
- `core/task_queue/workflow_executor.py` -- transitions task to DEFERRED
- `core/task_queue/scheduler.py`, `parallel.py`, `task_selector.py` -- schedule DEFERRED tasks
- `core/task_queue/workflows/illustrate_and_publish.py` -- returns "deferred" when budget exhausted

## Related

- [Task Queue Budget Tracking](../data-pipeline/task-queue-budget-tracking.md) -- broader budget management
- [Three-Layer Rate Limiting](../async-python/three-layer-rate-limiting.md) -- the rate limits that trigger deferral
- [Task Queue Interruption Recovery](../../solutions/workflow-reliability/task-queue-interruption-recovery.md) -- crash recovery for interrupted tasks
