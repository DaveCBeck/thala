---
name: task-queue-budget-tracking
title: "Task Queue with Budget Tracking: Orchestrated Workflow Processing with Cost Control"
date: 2026-01-28
category: data-pipeline
applicability:
  - "Long-running workflow orchestration with cost constraints"
  - "Multi-type task queue with checkpoint/resume capability"
  - "Budget-aware batch processing with adaptive scheduling"
components: [async_task, configuration, workflow_graph]
complexity: complex
verified_in_production: true
related_solutions: []
tags: [task-queue, budget, langsmith, checkpoint, resume, fcntl, round-robin, cost-tracking, orchestration]
---

# Task Queue with Budget Tracking: Orchestrated Workflow Processing with Cost Control

## Intent

Provide a file-based task queue with LangSmith cost tracking, PID-based crash recovery, round-robin category scheduling, and adaptive pacing based on budget consumption rate.

## Motivation

Running long-duration LLM workflows (literature reviews, web research) requires:

1. **Budget control**: Prevent runaway costs by tracking spend via LangSmith and enforcing limits
2. **Crash recovery**: Resume interrupted workflows from the last checkpoint instead of restarting
3. **Fair scheduling**: Rotate across categories to ensure balanced topic coverage
4. **Concurrent access**: Allow CLI and cron jobs to safely access the queue simultaneously

**The Solution:**
```
┌─────────────────────────────────────────────────────────────────────┐
│  Task Queue Architecture                                             │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │ queue.json   │   │ current_work │   │ cost_cache.json          │ │
│  │ - tasks[]    │   │ .json        │   │ - periods: {month: cost} │ │
│  │ - categories │   │ - active[]   │   │ - last_sync              │ │
│  │ - concurrency│   │ - pids{}     │   └──────────────────────────┘ │
│  └──────┬───────┘   └──────┬───────┘                                │
│         │                  │                                         │
│         ▼                  ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     TaskQueueManager                          │   │
│  │  - fcntl.flock() for cross-process coordination              │   │
│  │  - atomic writes via temp file + rename                       │   │
│  │  - round-robin category selection                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         ▼                    ▼                    ▼                  │
│  ┌─────────────┐   ┌─────────────────┐   ┌────────────────────┐    │
│  │ Checkpoint  │   │ Budget Tracker  │   │ Workflow Runner    │    │
│  │ Manager     │   │ (LangSmith)     │   │ (async + dispatch) │    │
│  └─────────────┘   └─────────────────┘   └────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Applicability

Use this pattern when:
- Running expensive LLM workflows that need cost guardrails
- Workflows can crash/restart and need checkpoint recovery
- Multiple task types need fair scheduling across categories
- CLI and background processes need concurrent queue access

Do NOT use this pattern when:
- Simple one-off scripts (use direct workflow invocation)
- Real-time/low-latency requirements (file locking adds overhead)
- No budget constraints or crash recovery needs

## Structure

```
core/task_queue/
├── schemas.py           # TypedDict definitions (Task, Checkpoint, etc.)
├── queue_manager.py     # Queue CRUD with fcntl locking
├── checkpoint_manager.py # PID-based resume tracking
├── budget_tracker.py    # LangSmith cost aggregation
├── runner.py            # Workflow orchestration
├── cli.py               # Rich-based CLI interface
├── paths.py             # Centralized path constants
├── pricing.py           # Model cost lookups
└── workflows/           # Workflow implementations
    ├── base.py          # Abstract workflow interface
    ├── lit_review_full.py
    ├── web_research.py
    └── publish_series.py
```

## Implementation

### Step 1: File-Based Queue with fcntl Locking

```python
# core/task_queue/queue_manager.py

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

class TaskQueueManager:
    """Manages the task queue with safe concurrent access."""

    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.queue_file = queue_dir / "queue.json"
        self.lock_file = queue_dir / "queue.lock"
        queue_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self):
        """Acquire exclusive lock on queue file."""
        self.lock_file.touch(exist_ok=True)
        lock_fd = open(self.lock_file, "w")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()

    def _write_queue(self, queue: dict) -> None:
        """Write queue atomically via temp file + rename."""
        temp_file = self.queue_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(queue, f, indent=2)
        temp_file.rename(self.queue_file)  # Atomic on POSIX

    def add_task(self, task: dict) -> str:
        """Add task with lock protection."""
        with self._lock():
            queue = self._read_queue()
            queue["topics"].append(task)
            self._write_queue(queue)
        return task["id"]
```

### Step 2: Budget Tracking via LangSmith

```python
# core/task_queue/budget_tracker.py

from langsmith import Client
from datetime import datetime, timezone

CACHE_TTL_HOURS = 1.0
BudgetAction = Literal["pause", "slowdown", "warn", "ok"]

class BudgetTracker:
    """Track LLM costs via LangSmith and enforce budget limits."""

    def __init__(self, queue_dir: Path):
        self.cost_cache_file = queue_dir / "cost_cache.json"
        self.monthly_budget = float(os.getenv("THALA_MONTHLY_BUDGET", "100.0"))
        self.langsmith_project = os.getenv("THALA_QUEUE_PROJECT", "thala-queue")
        self._client = None

    @property
    def client(self):
        """Lazy-load LangSmith client."""
        if self._client is None:
            self._client = Client()
        return self._client

    def get_current_month_cost(self, force_refresh: bool = False) -> float:
        """Get total cost for current month with 1hr cache."""
        period = self._get_current_period()
        cache = self._read_cache()

        # Return cached if valid
        if not force_refresh and self._is_cache_valid(cache.get("periods", {}).get(period)):
            return cache["periods"][period]["total_cost_usd"]

        # Query LangSmith for root traces (costs are aggregated)
        start_of_month = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        total_cost = 0.0
        runs = self.client.list_runs(
            project_name=self.langsmith_project,
            start_time=start_of_month,
            is_root=True,  # Only root traces (child costs aggregated)
        )

        for run in runs:
            if run.total_cost:
                total_cost += float(run.total_cost)

        # Update cache
        self._update_cache(period, total_cost)
        return total_cost

    def get_budget_status(self) -> dict:
        """Get current budget status with recommended action."""
        current_cost = self.get_current_month_cost()
        percent_used = (current_cost / self.monthly_budget) * 100

        # Determine action based on thresholds
        if percent_used >= 100:
            action = "pause"
        elif percent_used >= 90:
            action = "slowdown"
        elif percent_used >= 75:
            action = "warn"
        else:
            action = "ok"

        return {
            "current_cost": current_cost,
            "monthly_budget": self.monthly_budget,
            "percent_used": percent_used,
            "action": action,
        }
```

### Step 3: Checkpoint/Resume with PID Detection

```python
# core/task_queue/checkpoint_manager.py

import os

class CheckpointManager:
    """Manage workflow checkpoints for resume capability."""

    def __init__(self, queue_dir: Path):
        self.current_work_file = queue_dir / "current_work.json"

    def start_work(self, task_id: str, task_type: str, langsmith_run_id: str):
        """Record work started with PID lock."""
        work = self._read_current_work()

        checkpoint = {
            "task_id": task_id,
            "task_type": task_type,
            "langsmith_run_id": langsmith_run_id,
            "phase": "start",
            "started_at": datetime.utcnow().isoformat(),
            "counters": {},
        }

        work["active_tasks"].append(checkpoint)
        work["process_locks"][task_id] = str(os.getpid())  # PID lock
        self._write_current_work(work)

    def get_incomplete_work(self) -> list[dict]:
        """Get work items where owning process is dead.

        Uses os.kill(pid, 0) to check if process exists without sending signal.
        """
        work = self._read_current_work()
        incomplete = []

        for checkpoint in work["active_tasks"]:
            task_id = checkpoint["task_id"]
            lock_pid = work["process_locks"].get(task_id)

            if lock_pid:
                try:
                    os.kill(int(lock_pid), 0)  # Check existence only
                    continue  # Process alive, skip
                except (OSError, ValueError):
                    pass  # Process dead, this is incomplete

            incomplete.append(checkpoint)

        return incomplete

    def update_checkpoint(self, task_id: str, phase: str, **counters):
        """Update checkpoint with phase and optional counters."""
        work = self._read_current_work()
        for cp in work["active_tasks"]:
            if cp["task_id"] == task_id:
                cp["phase"] = phase
                cp["last_checkpoint_at"] = datetime.utcnow().isoformat()
                cp["counters"].update(counters)
                break
        self._write_current_work(work)
```

### Step 4: Round-Robin Category Scheduling

```python
# core/task_queue/queue_manager.py (continued)

def get_next_eligible_task(self) -> Optional[dict]:
    """Select next task via round-robin category rotation."""
    with self._lock():
        queue = self._read_queue()

        # Check concurrency constraints
        if not self._can_start_new_task(queue):
            return None

        pending = [t for t in queue["topics"] if t["status"] == "pending"]
        if not pending:
            return None

        categories = queue["categories"]
        last_idx = queue["last_category_index"]

        # Try each category in round-robin order
        for offset in range(len(categories)):
            cat_idx = (last_idx + 1 + offset) % len(categories)
            category = categories[cat_idx]

            # Find highest priority task in this category
            cat_tasks = [t for t in pending if t["category"] == category]
            if cat_tasks:
                # Sort by priority (desc), then created_at (asc)
                cat_tasks.sort(key=lambda t: (-t["priority"], t["created_at"]))

                # Update round-robin index
                queue["last_category_index"] = cat_idx
                self._write_queue(queue)
                return cat_tasks[0]

        # Fallback: highest priority overall
        pending.sort(key=lambda t: (-t["priority"], t["created_at"]))
        return pending[0]
```

### Step 5: Adaptive Stagger Based on Budget Pace

```python
# core/task_queue/budget_tracker.py (continued)

def get_adaptive_stagger_hours(self, base_hours: float = 36.0) -> float:
    """Adjust stagger based on budget consumption rate.

    If under budget pace → speed up (reduce stagger)
    If over budget pace → slow down (increase stagger)
    """
    status = self.get_budget_status()

    # Expected percent based on day of month
    day_of_month = datetime.now(timezone.utc).day
    expected_percent = (day_of_month / 30) * 100

    # Ratio of actual to expected usage
    usage_ratio = status["percent_used"] / expected_percent if expected_percent > 0 else 1.0

    # Adjust stagger based on ratio
    if usage_ratio < 0.5:
        return base_hours * 0.5   # Way under budget - speed up
    elif usage_ratio < 0.8:
        return base_hours * 0.75  # Slightly under
    elif usage_ratio <= 1.2:
        return base_hours         # On track
    elif usage_ratio <= 1.5:
        return base_hours * 1.5   # Slightly over - slow down
    else:
        return base_hours * 2.0   # Way over - significant slowdown
```

## Complete Example

```python
# Running the queue loop with all components

from core.task_queue import (
    TaskQueueManager,
    CheckpointManager,
    BudgetTracker,
    run_task_workflow,
)

async def run_queue_loop():
    """Process tasks with budget and checkpoint integration."""
    queue_manager = TaskQueueManager()
    checkpoint_mgr = CheckpointManager()
    budget_tracker = BudgetTracker()

    while True:
        # 1. Check for incomplete work (crashed processes)
        incomplete = checkpoint_mgr.get_incomplete_work()
        if incomplete:
            checkpoint = incomplete[0]
            task = queue_manager.get_task(checkpoint["task_id"])

            # Resume from checkpoint
            await run_task_workflow(
                task, queue_manager, checkpoint_mgr, budget_tracker,
                resume_from=checkpoint,
            )
            continue

        # 2. Get next eligible task (round-robin + concurrency check)
        task = queue_manager.get_next_eligible_task()
        if not task:
            await asyncio.sleep(300)  # 5 min check interval
            continue

        # 3. Check budget
        should_proceed, reason = budget_tracker.should_proceed()
        if not should_proceed:
            logger.warning(f"Budget pause: {reason}")
            await asyncio.sleep(3600)  # Wait 1 hour
            continue

        # 4. Run the task
        await run_task_workflow(
            task, queue_manager, checkpoint_mgr, budget_tracker,
        )

        # 5. Adaptive delay before next task
        config = queue_manager.get_concurrency_config()
        if config["mode"] == "stagger_hours":
            adaptive_hours = budget_tracker.get_adaptive_stagger_hours(
                config["stagger_hours"]
            )
            await asyncio.sleep(adaptive_hours * 3600)
```

## Consequences

### Benefits

- **Cost control**: LangSmith-based tracking with pause/slowdown at thresholds
- **Crash recovery**: PID-based detection resumes from last checkpoint
- **Fair scheduling**: Round-robin ensures all categories get processed
- **Safe concurrency**: fcntl locking allows CLI + cron to coexist
- **Adaptive pacing**: Budget consumption rate adjusts task spacing
- **Workflow isolation**: Dedicated LangSmith project separates queue costs from dev testing

### Trade-offs

- **File I/O overhead**: Every operation requires file read/write with locking
- **Single-machine**: fcntl doesn't work across machines (use Redis for distributed)
- **Cache staleness**: 1hr TTL means costs can be up to 1 hour behind
- **No real-time updates**: Polling-based, not event-driven

### Alternatives

- **Redis + Celery**: For distributed task processing
- **PostgreSQL + pg_advisory_lock**: For database-backed queues
- **AWS SQS**: For cloud-native queue management
- **Manual cost tracking**: If LangSmith isn't available

## Related Patterns

- [LangSmith Workflow Tracing](../../patterns/llm-interaction/langsmith-workflow-tracing.md) - Tracing infrastructure
- [Batch API Cost Optimization](../../patterns/llm-interaction/batch-api-cost-optimization.md) - 50% savings for Claude
- [Model Tier Optimization](../../solutions/llm-issues/model-tier-optimization.md) - Tier selection

## Known Uses in Thala

- `core/task_queue/queue_manager.py`: File-based queue with fcntl locking
- `core/task_queue/budget_tracker.py`: LangSmith cost aggregation
- `core/task_queue/checkpoint_manager.py`: PID-based resume tracking
- `core/task_queue/runner.py`: Workflow orchestration with adaptive delays
- `core/task_queue/cli.py`: Rich-based CLI for queue management
- `core/task_queue/workflows/lit_review_full.py`: Literature review workflow
- `core/task_queue/workflows/web_research.py`: Web research workflow

## References

- Commit: d3a94b9
- [LangSmith Python SDK](https://docs.smith.langchain.com/reference/python/sdk)
- [fcntl - Unix file locking](https://docs.python.org/3/library/fcntl.html)
