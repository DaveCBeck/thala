# Task Queue

Persistent queue infrastructure for managing long-running workflows with budget awareness and checkpoint/resume capability. Supports multiple workflow types via a registry pattern.

## Workflow Types

| Type | Pipeline | Use Case |
|------|----------|----------|
| `lit_review_full` | lit_review → enhance → evening_reads → illustrate → spawn_publish | Academic research with full enhancement and publishing |
| `web_research` | deep_research → evening_reads | Web-based research and article generation |
| `publish_series` | checking → publishing | Schedule-aware draft publishing to Substack |

## Usage

### CLI Commands

```bash
# Add literature review task (default)
python -m core.task_queue.cli add "quantum computing applications" \
  -c technology --priority high --quality standard

# Add web research task
python -m core.task_queue.cli add "AI impact on employment 2025" \
  -c technology --type web_research --quality standard

# Show status (budget, queue, next task)
python -m core.task_queue.cli status

# List tasks
python -m core.task_queue.cli list --status pending

# Configure concurrency
python -m core.task_queue.cli config --mode stagger_hours --stagger-hours 24

# Start/stop daemon
python -m core.task_queue.cli start
python -m core.task_queue.cli stop
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

### publish_series Fields

| Field | Type | Description |
|-------|------|-------------|
| `base_date` | str | ISO datetime (Monday 3pm local) |
| `items` | list[PublishItem] | The 5 items to publish |
| `source_task_id` | str | ID of lit_review_full task that spawned this |

### PublishItem Structure

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | "overview", "lit_review", "deep_dive_1", etc. |
| `title` | str | Article title |
| `path` | str | Path to illustrated markdown file |
| `day_offset` | int | Days from base_date to publish |
| `audience` | str | "everyone" or "only_paid" |
| `published` | bool | Has this item been published? |
| `draft_id` | str | Substack draft ID once created |
| `draft_url` | str | URL to draft in Substack |

## Publication Schedule

When a `lit_review_full` task completes, it spawns a `publish_series` task with this schedule:

| Day Offset | Item | Audience |
|------------|------|----------|
| 0 | Overview | everyone |
| +1 | Lit Review | only_paid |
| +4 | Deep Dive 1 | everyone |
| +7 | Deep Dive 2 | everyone |
| +11 | Deep Dive 3 | everyone |

The base date is calculated as the next Monday at 3pm local time that doesn't conflict with existing `publish_series` tasks in the same category.

## File Storage

All local state is stored under `.thala/`:

- `.thala/queue/queue.json` - Persistent task queue (LLM-editable JSON)
- `.thala/queue/current_work.json` - Active work with checkpoints
- `.thala/queue/cost_cache.json` - Monthly cost aggregations (1hr TTL)
- `.thala/queue/publications.json` - Category → Substack publication mapping
- `.thala/output/` - Generated reports and article series
- `.thala/.substack-cookies.json` - Substack authentication cookies

### Publications Config (Source of Truth for Categories)

The `.thala/queue/publications.json` file is the **source of truth for categories**. The top-level keys define which categories exist in the system, and the values map each category to its Substack publication:

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

Categories are loaded from this file when the queue manager initializes. The `publish_series` workflow also uses this config to route drafts to the correct Substack publication.

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
    "publish_series": PublishSeriesWorkflow,
    "web_research": WebResearchWorkflow,
}
```

Each workflow defines its own checkpoint phases.

### Zero-Cost Workflows

Some workflows (like `publish_series`) make no LLM calls and can be marked as zero-cost:

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
python -m core.task_queue.cli config --mode stagger_hours --stagger-hours 24

# Via API
queue.set_concurrency(mode="stagger_hours", stagger_hours=24.0)
```

## Related Modules

- `workflows.research.academic_lit_review` - Literature review workflow
- `workflows.research.web_research` - Web research workflow
- `workflows.enhance` - Enhancement (supervision + editing)
- `workflows.output.evening_reads` - Article series generation
- `workflows.output.illustrate` - Document illustration with images
- `utils.substack_publish` - Substack draft/publish API
