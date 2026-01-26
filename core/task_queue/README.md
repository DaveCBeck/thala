# Task Queue

Persistent queue infrastructure for managing long-running literature review workflows with budget awareness and checkpoint/resume capability.

## Usage

### CLI Commands

```bash
# Add task to queue
python -m core.task_queue.cli add "quantum computing applications" \
  -c technology --priority high --quality standard

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

# Initialize manager
queue = TaskQueueManager()

# Add task
task_id = queue.add_task(
    topic="quantum computing applications",
    category=TaskCategory.TECHNOLOGY,
    priority=TaskPriority.HIGH,
    quality="standard",
    language="en",
    date_range=(2020, 2026),
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

# Run next eligible task
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

# Start work
checkpoint.start_work(topic_id, langsmith_run_id)

# Update checkpoint during workflow
checkpoint.update_checkpoint(
    topic_id,
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

## Input/Output

### Task Structure
| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID |
| `topic` | str | Main topic text |
| `category` | str | philosophy, science, technology, society, culture |
| `priority` | int | 1-4 (low, normal, high, urgent) |
| `status` | str | pending, in_progress, paused, completed, failed |
| `quality` | str | test, quick, standard, comprehensive, high_quality |
| `research_questions` | list[str] | Optional pre-defined questions |
| `langsmith_run_id` | str | For cost attribution and trace lookup |

### File Storage
- `topic_queue/queue.json` - Persistent task queue (LLM-editable JSON)
- `topic_queue/current_work.json` - Active work with checkpoints
- `topic_queue/cost_cache.json` - Monthly cost aggregations (1hr TTL)
- `.outputs/` - Generated literature reviews and article series

## Architecture

### Key Components

**TaskQueueManager**: Thread-safe queue management with fcntl-based file locking. Implements:
- Round-robin category selection for thematic diversity
- Priority within category
- Flexible concurrency (max_concurrent or stagger_hours modes)
- Atomic writes via temp file + rename

**CheckpointManager**: Workflow progress tracking with PID-based process locking. Phases:
- `discovery` - Research questions + keyword search
- `diffusion` - Citation network expansion
- `processing` - Paper summarization
- `clustering` - Thematic grouping
- `synthesis` - Literature review generation

**BudgetTracker**: LangSmith cost aggregation with adaptive behavior:
- Queries `list_runs()` for monthly totals (root traces only)
- 1-hour cache TTL to minimize API calls
- Three actions: `pause` (100%), `slowdown` (90%), `warn` (75%)
- Adaptive stagger calculation based on budget pace

**Runner**: Orchestrates workflows with checkpoint callbacks and budget checks. Uses dedicated LangSmith project for budget isolation from manual testing.

### Concurrency Control

**stagger_hours** mode (default): Minimum time between task starts
- Prevents overwhelming Semantic Scholar API
- Adaptive stagger adjusts based on budget pace
- Default: 36 hours

**max_concurrent** mode: Maximum simultaneous tasks
- Simple parallel execution
- No built-in API rate limiting

### Round-Robin Scheduling

Tasks selected using category rotation with priority as tiebreaker:
1. Next category in rotation
2. Highest priority task in that category
3. Oldest task if priorities equal
4. Falls back to highest priority overall if rotation empty

## Configuration

### Environment Variables

```bash
# Budget configuration
THALA_MONTHLY_BUDGET=100.0        # USD per month
THALA_BUDGET_ACTION=pause          # pause, slowdown, or warn

# LangSmith integration
THALA_QUEUE_PROJECT=thala-queue    # Dedicated project for queue runs
LANGSMITH_API_KEY=...              # Required for cost tracking
LANGSMITH_PROJECT=thala-queue      # Set automatically by runner
```

### Concurrency Tuning

```python
# Via CLI
python -m core.task_queue.cli config \
  --mode stagger_hours --stagger-hours 24

# Via API
queue.set_concurrency(
    mode="stagger_hours",
    stagger_hours=24.0,
)
```

### Categories

Edit categories in `topic_queue/queue.json` or via API:

```python
queue.set_categories([
    "philosophy",
    "science",
    "technology",
    "society",
    "culture",
    "custom_category",
])
```

## Related Modules

- `workflows.research.academic_lit_review` - Literature review workflow
- `workflows.output.evening_reads` - Article series generation
- `core.semantic_scholar` - Paper search and metadata
- `core.diffusion` - Citation network expansion
- `core.processing` - Paper processing and summarization
- `core.clustering` - Thematic clustering
- `core.synthesis` - Literature review synthesis
