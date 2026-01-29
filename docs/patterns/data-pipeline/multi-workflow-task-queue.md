---
name: multi-workflow-task-queue
title: "Multi-Workflow Task Queue Architecture: Registry-Based Polymorphic Workflow Dispatch"
date: 2026-01-29
category: data-pipeline
applicability:
  - "Task queues supporting multiple distinct workflow types"
  - "Workflows with different checkpoint phases and resource requirements"
  - "Parent-child task pipelines with spawn semantics"
components: [async_task, workflow_graph, configuration]
complexity: complex
verified_in_production: true
related_solutions: []
tags: [task-queue, workflow, registry, polymorphism, checkpoint, spawn, zero-cost, abstract-base, multi-stage]
---

# Multi-Workflow Task Queue Architecture: Registry-Based Polymorphic Workflow Dispatch

## Intent

Provide a polymorphic task queue that dispatches to different workflow implementations based on task type, with workflow-specific checkpoint phases, zero-cost bypass, and parent-child task spawning.

## Motivation

A task queue initially built for a single workflow type (e.g., literature reviews) needs to support additional workflow types:

1. **Different pipelines**: Academic research vs web research vs publishing have distinct phases
2. **Different resource profiles**: Some workflows consume LLM budget, others are zero-cost (API-only)
3. **Multi-stage pipelines**: One workflow may spawn child tasks for downstream processing
4. **Checkpoint variance**: Each workflow has unique resumption points

**The Solution:**
```
+-----------------------------------------------------------------------+
|  Multi-Workflow Task Queue Architecture                                |
|                                                                        |
|  +------------------+                                                  |
|  | WORKFLOW_REGISTRY| <-- {task_type: WorkflowClass}                   |
|  +--------+---------+                                                  |
|           |                                                            |
|           v                                                            |
|  +------------------+     +---------------------+                      |
|  | get_workflow()   |---->| BaseWorkflow (ABC)  |                      |
|  +------------------+     | - task_type         |                      |
|                           | - phases[]          |                      |
|                           | - is_zero_cost      |                      |
|                           | - run()             |                      |
|                           | - save_outputs()    |                      |
|                           +----------+----------+                      |
|                                      |                                 |
|            +-------------------------+-------------------------+       |
|            |                         |                         |       |
|  +---------v--------+    +-----------v---------+   +-----------v----+  |
|  | LitReviewFull    |    | WebResearch         |   | PublishSeries  |  |
|  | phases:          |    | phases:             |   | phases:        |  |
|  | - lit_review     |    | - research          |   | - checking     |  |
|  | - supervision    |    | - evening_reads     |   | - publishing   |  |
|  | - editing        |    | - saving            |   | - complete     |  |
|  | - evening_reads  |    | - complete          |   | is_zero_cost:  |  |
|  | - illustrate     |    +---------------------+   | True           |  |
|  | - spawn_publish  |                              +----------------+  |
|  | - saving         |---spawn---> PublishSeriesTask                    |
|  | - complete       |                                                  |
|  +------------------+                                                  |
+-----------------------------------------------------------------------+
```

## Applicability

Use this pattern when:
- Task queue must support multiple fundamentally different workflow types
- Workflows have distinct checkpoint phases for resumption
- Some workflows should bypass cost/budget checks (zero-cost)
- Workflows need to spawn child tasks for downstream processing
- Generic counter storage must replace hardcoded workflow-specific fields

Do NOT use this pattern when:
- Single workflow type is sufficient (use direct implementation)
- Workflows are trivial variations (use configuration instead)
- No checkpoint/resume requirements exist

## Structure

```
core/task_queue/
+-- schemas.py               # TaskType enum, Task union, generic checkpoint
+-- workflows/
|   +-- __init__.py          # WORKFLOW_REGISTRY, get_workflow(), get_phases()
|   +-- base.py              # BaseWorkflow abstract class
|   +-- lit_review_full.py   # Academic literature review workflow
|   +-- web_research.py      # Web research workflow
|   +-- publish_series.py    # Schedule-aware publishing (zero-cost)
+-- checkpoint_manager.py    # Workflow-aware phase tracking
+-- runner.py                # Registry-based dispatch
```

## Implementation

### Step 1: Define Abstract Workflow Interface

```python
# core/task_queue/workflows/base.py

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

class BaseWorkflow(ABC):
    """Abstract base class for workflow implementations.

    Subclasses must implement:
    - task_type: Identifier matching TaskType enum
    - phases: Ordered list of checkpoint phases
    - run(): Execute the workflow
    - save_outputs(): Save results to disk
    """

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return the task type identifier.

        Must match the TaskType enum value and the key used
        in WORKFLOW_REGISTRY.
        """
        pass

    @property
    @abstractmethod
    def phases(self) -> list[str]:
        """Return ordered list of workflow phases for checkpointing.

        These phases are used by CheckpointManager to track progress
        and enable resumption from the last completed phase.

        The last phase should typically be "complete".
        """
        pass

    @abstractmethod
    async def run(
        self,
        task: dict[str, Any],
        checkpoint_callback: Callable[[str], None],
        resume_from: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Execute the workflow.

        Args:
            task: Task data from queue (serialized TaskType)
            checkpoint_callback: Call with phase name to update progress
            resume_from: Optional checkpoint dict to resume from

        Returns:
            Dict with at minimum:
            - status: "success", "partial", or "failed"
            - Any workflow-specific outputs
        """
        pass

    @abstractmethod
    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """Save workflow outputs to disk.

        Returns:
            Dict mapping output names to file paths
        """
        pass

    # Optional: Override for zero-cost workflows
    @property
    def is_zero_cost(self) -> bool:
        """Return True to bypass budget checks."""
        return False

    @property
    def bypass_concurrency(self) -> bool:
        """Return True to bypass stagger_hours/max_concurrent limits."""
        return False

    # Helper methods for all workflows
    def get_task_identifier(self, task: dict[str, Any]) -> str:
        """Get a human-readable identifier for the task."""
        for field in ["topic", "query", "title"]:
            if field in task:
                return task[field][:50]
        return task.get("id", "unknown")[:8]

    def slugify(self, text: str, max_length: int = 50) -> str:
        """Convert text to a filesystem-safe slug."""
        return text[:max_length].replace(" ", "_").replace("/", "-")
```

### Step 2: Create Workflow Registry

```python
# core/task_queue/workflows/__init__.py

from typing import TYPE_CHECKING
from .base import BaseWorkflow
from .lit_review_full import LitReviewFullWorkflow
from .publish_series import PublishSeriesWorkflow
from .web_research import WebResearchWorkflow

# Registry mapping task_type -> workflow class
WORKFLOW_REGISTRY: dict[str, type[BaseWorkflow]] = {
    "lit_review_full": LitReviewFullWorkflow,
    "publish_series": PublishSeriesWorkflow,
    "web_research": WebResearchWorkflow,
}

# Default workflow type for backward compatibility
DEFAULT_WORKFLOW_TYPE = "lit_review_full"


def get_workflow(task_type: str) -> BaseWorkflow:
    """Get workflow instance for a task type.

    Args:
        task_type: The workflow type identifier

    Returns:
        Instantiated workflow object

    Raises:
        ValueError: If task_type is not registered
    """
    if task_type not in WORKFLOW_REGISTRY:
        available = ", ".join(WORKFLOW_REGISTRY.keys())
        raise ValueError(f"Unknown task type: {task_type}. Available: {available}")
    return WORKFLOW_REGISTRY[task_type]()


def get_phases(task_type: str) -> list[str]:
    """Get checkpoint phases for a workflow type."""
    return get_workflow(task_type).phases


def get_available_types() -> list[str]:
    """Get list of available workflow types."""
    return list(WORKFLOW_REGISTRY.keys())
```

### Step 3: Implement Workflow-Specific Phases

```python
# core/task_queue/workflows/lit_review_full.py

class LitReviewFullWorkflow(BaseWorkflow):
    """Full literature review with enhancement and article series."""

    @property
    def task_type(self) -> str:
        return "lit_review_full"

    @property
    def phases(self) -> list[str]:
        return [
            "lit_review",      # Academic literature review generation
            "supervision",     # Enhancement loops
            "editing",         # Structural editing
            "evening_reads",   # Article series generation
            "illustrate",      # Add images to all articles
            "spawn_publish",   # Create publish_series task
            "saving",          # Output to disk
            "complete",
        ]

    async def run(self, task, checkpoint_callback, resume_from=None):
        # ... implementation ...
        checkpoint_callback("lit_review")
        # ... run lit review ...

        checkpoint_callback("supervision")
        # ... run enhancement ...

        checkpoint_callback("spawn_publish")
        publish_task_id = self._spawn_publish_task(task, illustrated_paths)

        return {"status": "success", "publish_task_id": publish_task_id}
```

```python
# core/task_queue/workflows/web_research.py

class WebResearchWorkflow(BaseWorkflow):
    """Web research with article series generation."""

    @property
    def task_type(self) -> str:
        return "web_research"

    @property
    def phases(self) -> list[str]:
        return [
            "research",        # Deep web research
            "evening_reads",   # Article series generation
            "saving",          # Output to disk
            "complete",
        ]
```

```python
# core/task_queue/workflows/publish_series.py

class PublishSeriesWorkflow(BaseWorkflow):
    """Schedule-aware draft publishing workflow."""

    @property
    def task_type(self) -> str:
        return "publish_series"

    @property
    def phases(self) -> list[str]:
        return ["checking", "publishing", "complete"]

    @property
    def is_zero_cost(self) -> bool:
        """This workflow makes no LLM calls, skip budget check."""
        return True

    @property
    def bypass_concurrency(self) -> bool:
        """Low-overhead publishing can run anytime, bypass stagger limits."""
        return True
```

### Step 4: Generic Counter Storage in Checkpoints

```python
# core/task_queue/schemas.py

class WorkflowCheckpoint(TypedDict):
    """Checkpoint data for workflow resumption.

    Now workflow-aware: task_type determines valid phases.
    Stores phase_outputs for resumption after interruption.
    """

    task_id: str           # Renamed from topic_id for genericity
    task_type: str         # Workflow type for phase validation
    langsmith_run_id: str
    phase: str             # Current phase (workflow-specific)
    phase_progress: dict   # Phase-specific progress data
    phase_outputs: dict    # Outputs from completed phases for resumption
    started_at: str
    last_checkpoint_at: str

    # Generic counters storage (workflow-specific)
    # For lit_review_full: papers_discovered, papers_processed, etc.
    # For web_research: sources_found, etc.
    counters: dict         # {counter_name: value}
```

### Step 5: Registry-Based Dispatch in Runner

```python
# core/task_queue/runner.py

from .workflows import get_workflow, DEFAULT_WORKFLOW_TYPE

async def run_task_workflow(
    task: Task,
    queue_manager: TaskQueueManager,
    checkpoint_mgr: CheckpointManager,
    budget_tracker: BudgetTracker,
    resume_from: Optional[WorkflowCheckpoint] = None,
) -> dict:
    """Run a workflow for any task type via registry dispatch."""

    # Get task type and workflow
    task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
    workflow = get_workflow(task_type)

    task_id = task["id"]
    task_identifier = workflow.get_task_identifier(task)

    # Mark as started
    queue_manager.mark_started(task_id, langsmith_run_id)
    checkpoint_mgr.start_work(task_id, task_type, langsmith_run_id)

    # Create checkpoint callback
    def checkpoint_callback(phase: str, **kwargs):
        checkpoint_mgr.update_checkpoint(task_id, phase, **kwargs)
        queue_manager.update_phase(task_id, phase)

    # Run the workflow
    result = await workflow.run(task, checkpoint_callback, resume_from)

    # Save outputs
    output_paths = workflow.save_outputs(task, result)
    result["output_paths"] = output_paths

    return result
```

### Step 6: Zero-Cost Workflow Budget Bypass

```python
# core/task_queue/runner.py (continued)

async def run_single_task(...):
    task = queue_manager.get_next_eligible_task()
    if not task:
        return None

    task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)

    # Check if zero-cost workflow (skip budget check)
    workflow = get_workflow(task_type)
    is_zero_cost = getattr(workflow, "is_zero_cost", False)

    if not is_zero_cost:
        # Check budget for workflows that incur costs
        should_proceed, reason = budget_tracker.should_proceed()
        if not should_proceed:
            logger.warning(f"Cannot proceed: {reason}")
            return None
    else:
        logger.info(f"Skipping budget check for zero-cost workflow: {task_type}")

    return await run_task_workflow(task, ...)
```

### Step 7: Spawn Child Tasks

```python
# core/task_queue/workflows/lit_review_full.py

def _spawn_publish_task(
    self,
    task: dict[str, Any],
    illustrated_paths: dict[str, str],
) -> str:
    """Create a publish_series task linked to parent."""
    from ..queue_manager import TaskQueueManager

    queue = TaskQueueManager()
    base_date = queue.find_next_available_monday(task["category"])

    # Build publish items with schedule:
    # Day 0: Overview, Day 1: Lit Review (paid), Day 4/7/11: Deep Dives
    items = [
        {
            "id": "overview",
            "title": f"Overview: {task['topic']}",
            "path": illustrated_paths.get("overview", ""),
            "day_offset": 0,
            "audience": "everyone",
            "published": False,
            "draft_id": None,
            "draft_url": None,
        },
        # ... more items ...
    ]

    return queue.add_task(
        task_type="publish_series",
        category=task["category"],
        priority=task["priority"],
        base_date=base_date.isoformat(),
        items=items,
        source_task_id=task["id"],  # Link to parent
    )
```

## Complete Example

```python
# Adding a new workflow type

# 1. Define the workflow
class MyCustomWorkflow(BaseWorkflow):
    @property
    def task_type(self) -> str:
        return "my_custom"

    @property
    def phases(self) -> list[str]:
        return ["step1", "step2", "step3", "complete"]

    @property
    def is_zero_cost(self) -> bool:
        return False  # Uses LLM budget

    async def run(self, task, checkpoint_callback, resume_from=None):
        # Track completed phases for resumption
        completed_phases = set()
        phase_outputs = {}
        if resume_from:
            completed_phases = self._get_completed_phases(resume_from)
            phase_outputs = resume_from.get("phase_outputs", {})

        if "step1" in completed_phases:
            result1 = phase_outputs.get("result1")
        else:
            checkpoint_callback("step1")
            result1 = await do_step1(task)
            checkpoint_callback("step1", phase_outputs={"result1": result1})

        if "step2" in completed_phases:
            result2 = phase_outputs.get("result2")
        else:
            checkpoint_callback("step2")
            result2 = await do_step2(result1)
            checkpoint_callback("step2", phase_outputs={"result1": result1, "result2": result2})

        checkpoint_callback("step3")
        final = await do_step3(result2)

        return {"status": "success", "output": final}

    def save_outputs(self, task, result):
        output_path = self.get_output_dir() / f"custom_{task['id']}.json"
        with open(output_path, "w") as f:
            json.dump(result, f)
        return {"output": str(output_path)}


# 2. Register in __init__.py
WORKFLOW_REGISTRY["my_custom"] = MyCustomWorkflow


# 3. Add task type to schemas.py
class TaskType(Enum):
    LIT_REVIEW_FULL = "lit_review_full"
    WEB_RESEARCH = "web_research"
    PUBLISH_SERIES = "publish_series"
    MY_CUSTOM = "my_custom"  # New type


# 4. Create task schema
class MyCustomTask(TypedDict):
    id: str
    task_type: str  # "my_custom"
    # ... custom fields ...


# 5. Add to Task union
Task = Union[TopicTask, WebResearchTask, PublishSeriesTask, MyCustomTask]
```

## Consequences

### Benefits

- **Polymorphic dispatch**: Single runner handles multiple workflow types cleanly
- **Independent phases**: Each workflow defines its own checkpoint sequence
- **Zero-cost bypass**: Non-LLM workflows skip budget checks automatically
- **Concurrency bypass**: Low-overhead workflows skip stagger/concurrency limits
- **Phase resumption**: Store phase_outputs to skip completed phases on resume
- **Parent-child linking**: Multi-stage pipelines via source_task_id
- **Generic counters**: No hardcoded fields, workflows define their own metrics
- **Backward compatibility**: DEFAULT_WORKFLOW_TYPE preserves existing behavior

### Trade-offs

- **Registry maintenance**: New workflows require registration in multiple places
- **Schema coupling**: Task union types must be updated for new task types
- **Phase divergence**: No enforcement that phases match workflow logic
- **Import cycles**: Workflow implementations must import dependencies lazily

### Alternatives

- **Configuration-based workflows**: Use YAML/JSON to define phases (less flexible)
- **Plugin architecture**: Dynamic discovery of workflow modules (more complex)
- **Single workflow with flags**: Conditional logic based on task fields (messy)

## Related Patterns

- [Task Queue with Budget Tracking](./task-queue-budget-tracking.md) - Budget and checkpoint foundation
- [LangGraph Workflow Architecture](../langgraph/langgraph-workflow-architecture.md) - Workflow graph patterns
- [Phased Pipeline Architecture](./phased-pipeline-architecture-gpu-queue.md) - Multi-phase processing

## Known Uses in Thala

- `core/task_queue/workflows/__init__.py`: Workflow registry with get_workflow()
- `core/task_queue/workflows/base.py`: BaseWorkflow abstract class
- `core/task_queue/workflows/lit_review_full.py`: 8-phase academic review workflow
- `core/task_queue/workflows/web_research.py`: 4-phase web research workflow
- `core/task_queue/workflows/publish_series.py`: Zero-cost publishing workflow
- `core/task_queue/runner.py`: Registry-based dispatch with zero-cost bypass
- `core/task_queue/checkpoint_manager.py`: Generic counter storage

## References

- Commit: 8529452
- [Registry Pattern](https://refactoring.guru/design-patterns/registry)
- [Template Method Pattern](https://refactoring.guru/design-patterns/template-method)
