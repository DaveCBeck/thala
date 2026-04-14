# Task Queue

Persistent queue infrastructure for managing long-running workflows with budget awareness and checkpoint/resume capability. Supports multiple workflow types via a registry pattern.

## Workflow Types

| Type | Pipeline | Use Case |
|------|----------|----------|
| `lit_review_full` | lit_review → enhance → evening_reads → save_and_spawn | Academic research with full enhancement and article generation |
| `web_research` | deep_research → evening_reads | Web-based research and article generation |
| `illustrate_and_export` | illustrate → batch export → rsync to VPS | Budget-aware illustration + batch export |

## Usage

### CLI Commands

```bash
# Add literature review task (default)
python3 -m core.task_queue.cli add "quantum computing applications" \
  -c technology --priority high --quality standard

# Add web research task
python3 -m core.task_queue.cli add "AI impact on employment 2025" \
  -c technology --type web_research --quality standard

# Show status (budget, queue, next task)
python3 -m core.task_queue.cli status

# List tasks
python3 -m core.task_queue.cli list --status pending

# Run up to 5 tasks in parallel with 3-minute stagger between starts
python3 -m core.task_queue.cli parallel --count 5 --stagger 3.0

# Configure concurrency
python3 -m core.task_queue.cli config --mode stagger_hours --stagger-hours 24

# Start/stop daemon
python3 -m core.task_queue.cli start
python3 -m core.task_queue.cli stop

# Pause/resume: block running workflows at their next checkpoint
# (see "Pause / Resume" section below)
python3 -m core.task_queue.cli pause --reason "peak-hours US"
python3 -m core.task_queue.cli resume
```

### Pause / Resume

Holds running workflows at their next natural checkpoint without killing
the process — useful for dodging peak hours on the Claude Max subscription
(Opus is more prone to truncation/compaction during US peak load) or any
time you want to step aside briefly without losing an in-flight iteration.

**Mechanism.** `pause` writes a flag file at `.thala/queue/paused`.
Runners call `await wait_if_paused()` at two hook points:

1. **After every incremental checkpoint save** (`IncrementalStateManager.save_progress`). In practice this means: after each supervision-loop iteration completes and is persisted, the runner blocks — the current integration always finishes and its state is on disk before anything stops.
2. **Before each task starts** (`run_task_workflow`). A paused runner won't begin selecting or launching new tasks.

Blocked callers poll the flag every 30 s. `resume` (or manually deleting
the flag file) lets them proceed within that window.

**Usage:**

```bash
# Pause — safe to run at any time; current LLM call will complete first
python3 -m core.task_queue.cli pause --reason "back at 22:00"

# Inspect
cat .thala/queue/paused          # shows timestamp + reason

# Resume
python3 -m core.task_queue.cli resume
```

**What pause does NOT do:**

- Does not interrupt an in-flight LLM call. An ongoing ~30-min Opus
  integration runs to completion; blocking happens afterwards.
- Does not affect tasks that don't call `save_progress` between phases
  (e.g. a single-shot workflow mid-call).
- Does not persist across `pause --reason` calls — subsequent pauses
  overwrite the marker.

**Automating off-peak runs.** Combine with `at` / cron:

```bash
# 13:00 UK: pause
echo 'cd /home/dave/thala && .venv/bin/python -m core.task_queue.cli pause --reason cron' | at 13:00

# 22:00 UK: resume
echo 'cd /home/dave/thala && .venv/bin/python -m core.task_queue.cli resume' | at 22:00
```

### Programmatic Usage

```python
from core.task_queue import TaskQueueManager, TaskCategory, TaskPriority

queue = TaskQueueManager()

# Add literature review task
task_id = queue.add_task(
    task_type="lit_review_full",
    topic="quantum computing applications",
    category=TaskCategory.TECHNOLOGY,
    priority=TaskPriority.HIGH,
    quality="standard",
    language="en",
    date_range=(2020, 2026),
)

# Add web research task
task_id = queue.add_task(
    task_type="web_research",
    query="AI impact on employment trends",
    category=TaskCategory.TECHNOLOGY,
    quality="standard",
)

# Get next eligible task (respects concurrency constraints)
task = queue.get_next_eligible_task()

# Update task status
queue.mark_started(task_id, langsmith_run_id="...")
queue.update_phase(task_id, "processing")
queue.mark_completed(task_id)
```

### Workflow Runner

```python
from core.task_queue.runner import run_single_task, run_queue_loop

# Run next eligible task (dispatches to correct workflow automatically)
result = await run_single_task()

# Continuous processing loop
await run_queue_loop(
    max_tasks=10,           # Stop after N tasks (None = unlimited)
    check_interval=300.0,   # Seconds between queue checks
)
```

### Budget Tracking

```python
from core.task_queue import BudgetTracker

tracker = BudgetTracker()

# Check budget status
status = tracker.get_budget_status()
print(f"Used: ${status['current_cost']:.2f} / ${status['monthly_budget']:.2f}")
print(f"Action: {status['action']}")  # pause, slowdown, warn, ok

# Should workflow proceed?
can_proceed, reason = tracker.should_proceed()

# Get adaptive stagger time (increases if over budget)
adaptive_hours = tracker.get_adaptive_stagger_hours(base_hours=36.0)
```

### Checkpointing

```python
from core.task_queue import CheckpointManager

checkpoint = CheckpointManager()

# Start work (now includes task_type)
checkpoint.start_work(task_id, task_type="lit_review_full", langsmith_run_id="...")

# Update checkpoint during workflow
checkpoint.update_checkpoint(
    task_id,
    phase="processing",
    papers_discovered=50,
    papers_processed=25,
)

# Resume incomplete work
incomplete = checkpoint.get_incomplete_work()
if incomplete:
    cp = incomplete[0]
    resume_phase = checkpoint.can_resume_from_phase(cp)
```

## Task Structure

### Common Fields (all types)

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID |
| `task_type` | str | Workflow type identifier |
| `category` | str | philosophy, science, technology, society, culture |
| `priority` | int | 1-4 (low, normal, high, urgent) |
| `status` | str | pending, in_progress, paused, completed, failed |
| `quality` | str | test, quick, standard, comprehensive, high_quality |
| `langsmith_run_id` | str | For cost attribution and trace lookup |

### lit_review_full Fields

| Field | Type | Description |
|-------|------|-------------|
| `topic` | str | Main topic text |
| `research_questions` | list[str] | Optional pre-defined questions |
| `language` | str | ISO 639-1 code |
| `date_range` | tuple[int, int] | (start_year, end_year) for papers |

### web_research Fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | Research query |
| `language` | str | Optional language override |

### illustrate_and_export Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_task_id` | str | Parent lit_review_full task ID |
| `topic` | str | Topic from parent task |
| `manifest_path` | str | Path to manifest.json |
| `items` | list[IllustrateExportItem] | Per-article progress tracking |
| `not_before` | str | ISO datetime — invisible until this time |
| `next_run_after` | str | ISO datetime for DEFERRED scheduling |

### IllustrateExportItem Structure

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | "overview", "deep_dive_1", etc. |
| `title` | str | Article title |
| `subtitle` | str | Short subtitle for Substack draft |
| `source_path` | str | Path to unillustrated markdown |
| `illustrated` | bool | Has illustration completed? |
| `illustrated_path` | str | Path to illustrated markdown (once done) |
| `exported` | bool | Whether article has been exported to batch folder |

## File Storage

All local state is stored under `.thala/`:

- `.thala/queue/queue.json` - Persistent task queue (LLM-editable JSON)
- `.thala/queue/current_work.json` - Active work with checkpoints
- `.thala/queue/cost_cache.json` - Monthly cost aggregations (1hr TTL)
- `.thala/queue/publications.json` - Category → publication mapping
- `.thala/output/` - Generated reports and article series
- `.thala/export/` - Batch-exported articles ready for rsync to VPS
- `.thala/state/pub_counters.json` - Per-publication sequential batch IDs

### Publications Config (Source of Truth for Categories)

The `.thala/queue/publications.json` file is the **source of truth for categories**. The top-level keys define which categories exist in the system, and the values map each category to its publication:

```json
{
  "philosophy": {
    "publication_url": "davecbeck.substack.com",
    "subdomain": "davecbeck"
  },
  "science": {
    "publication_url": "davecbeck.substack.com",
    "subdomain": "davecbeck"
  }
}
```

**To add a new category:** Add a new key to `.thala/queue/publications.json` with its publication config.

**To remove a category:** Delete the key from `.thala/queue/publications.json`.

Categories are loaded from this file when the queue manager initializes. The `illustrate_and_export` workflow uses this config to determine the publication slug for batch folders.

## Architecture

### Key Components

**TaskQueueManager**: Thread-safe queue management with fcntl-based file locking.
- Round-robin category selection for thematic diversity
- Priority within category
- Flexible concurrency (max_concurrent or stagger_hours modes)
- Atomic writes via temp file + rename

**CheckpointManager**: Workflow progress tracking with PID-based process locking.
- Dynamic phases per workflow type
- Generic counters storage
- Resume capability for crashed processes

**BudgetTracker**: LangSmith cost aggregation with adaptive behavior.
- Queries `list_runs()` for monthly totals (root traces only)
- 1-hour cache TTL to minimize API calls
- Three actions: `pause` (100%), `slowdown` (90%), `warn` (75%)

**Runner**: Dispatches tasks to workflow implementations via registry.
- Uses dedicated LangSmith project for budget isolation
- Automatic checkpoint callbacks
- Output saving handled by workflow class

### Workflow Registry

Workflows are registered in `core/task_queue/workflows/__init__.py`:

```python
WORKFLOW_REGISTRY = {
    "lit_review_full": LitReviewFullWorkflow,
    "web_research": WebResearchWorkflow,
    "illustrate_and_export": IllustrateAndExportWorkflow,
}
```

Each workflow defines its own checkpoint phases.

### Zero-Cost Workflows

Some workflows make no LLM calls and can be marked as zero-cost:

```python
class MyWorkflow(BaseWorkflow):
    @property
    def is_zero_cost(self) -> bool:
        return True  # Skip budget checks
```

Zero-cost workflows bypass budget checks and don't trigger stagger delays.

## Adding New Workflow Types

1. **Create workflow class** at `core/task_queue/workflows/my_workflow.py`:

```python
from .base import BaseWorkflow

class MyWorkflow(BaseWorkflow):
    @property
    def task_type(self) -> str:
        return "my_workflow"

    @property
    def phases(self) -> list[str]:
        return ["step1", "step2", "saving", "complete"]

    async def run(self, task, checkpoint_callback, resume_from=None):
        checkpoint_callback("step1")
        # ... do step 1 ...

        checkpoint_callback("step2")
        # ... do step 2 ...

        return {"status": "success", "output": result}

    def save_outputs(self, task, result):
        # Save to disk, return paths
        return {"output": "/path/to/output.md"}
```

2. **Register in `workflows/__init__.py`**:

```python
from .my_workflow import MyWorkflow

WORKFLOW_REGISTRY["my_workflow"] = MyWorkflow
```

3. **Add TypedDict to `schemas.py`** (if custom fields needed):

```python
class MyWorkflowTask(TypedDict):
    id: str
    task_type: str  # "my_workflow"
    my_field: str
    # ... common fields ...
```

4. **Update CLI** in `cli.py` (choices auto-update from registry).

## Configuration

### Environment Variables

```bash
# Budget configuration
THALA_MONTHLY_BUDGET=100.0        # USD per month
THALA_BUDGET_ACTION=pause          # pause, slowdown, or warn

# LangSmith integration
THALA_QUEUE_PROJECT=thala-queue    # Dedicated project for queue runs
LANGSMITH_API_KEY=...              # Required for cost tracking
```

### Concurrency Tuning

```bash
# Via CLI
python3 -m core.task_queue.cli config --mode stagger_hours --stagger-hours 24

# Via API
queue.set_concurrency(mode="stagger_hours", stagger_hours=24.0)
```

## Related Modules

- `workflows.research.academic_lit_review` - Literature review workflow
- `workflows.research.web_research` - Web research workflow
- `workflows.enhance` - Enhancement (supervision + editing)
- `workflows.output.evening_reads` - Article series generation
- `workflows.output.illustrate` - Document illustration with images
- `core.task_queue.workflows.shared.batch_export` - Batch export to staging directory
- `core.task_queue.workflows.shared.rsync_export` - Rsync batch to VPS
