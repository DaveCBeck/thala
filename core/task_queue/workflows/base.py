"""Base class for workflow implementations.

All workflow types must implement this interface to be compatible
with the task queue runner.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from ..paths import OUTPUT_DIR


class BaseWorkflow(ABC):
    """Abstract base class for workflow implementations.

    Subclasses must implement:
    - task_type: Identifier matching the TaskType enum
    - phases: Ordered list of checkpoint phases
    - run(): Execute the workflow
    - save_outputs(): Save results to disk

    Example:
        class MyWorkflow(BaseWorkflow):
            @property
            def task_type(self) -> str:
                return "my_workflow"

            @property
            def phases(self) -> list[str]:
                return ["step1", "step2", "complete"]

            async def run(self, task, checkpoint_callback, resume_from=None):
                checkpoint_callback("step1")
                # ... do step 1 ...
                checkpoint_callback("step2")
                # ... do step 2 ...
                return {"status": "success", "output": result}

            def save_outputs(self, task, result):
                # Save to disk, return paths
                return {"output": "/path/to/output.md"}
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
        *,
        flush_checkpoints: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
        """Execute the workflow.

        Args:
            task: Task data from queue (TopicTask, WebResearchTask, etc.)
            checkpoint_callback: Call with phase name to update progress.
                Signature: checkpoint_callback(phase: str, **kwargs)
            resume_from: Optional checkpoint dict to resume from.
                Contains: phase, phase_progress, counters, etc.
            flush_checkpoints: Optional async function to await all pending
                checkpoint writes. Call before operations that depend on
                checkpoints being persisted (e.g., clearing incremental state).

        Returns:
            Dict with at minimum:
            - status: "success", "partial", or "failed"
            - Any workflow-specific outputs (e.g., lit_review, series, etc.)

        Raises:
            Exception: On unrecoverable errors (will mark task as failed)
        """
        pass

    @abstractmethod
    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """Save workflow outputs to disk.

        Args:
            task: Task data from queue
            result: Result dict from run()

        Returns:
            Dict mapping output names to file paths
        """
        pass

    def get_task_identifier(self, task: dict[str, Any]) -> str:
        """Get a human-readable identifier for the task.

        Used in logging and output filenames.

        Args:
            task: Task data from queue

        Returns:
            Short identifier string (typically the topic/query truncated)
        """
        # Try common fields
        for field in ["topic", "query", "title"]:
            if field in task:
                return task[field][:50]
        return task.get("id", "unknown")[:8]

    @property
    def is_zero_cost(self) -> bool:
        """Return True if this workflow makes no LLM calls.

        Zero-cost workflows skip budget checks.
        Default is False (most workflows incur LLM costs).
        """
        return False

    @property
    def bypass_concurrency(self) -> bool:
        """Return True if this workflow should bypass concurrency limits.

        Bypass workflows can run regardless of stagger_hours or max_concurrent
        settings. Useful for low-overhead tasks like publishing that shouldn't
        block or be blocked by expensive workflows.

        Default is False (respects concurrency limits).
        """
        return False

    def get_output_dir(self) -> Path:
        """Get the output directory for this workflow.

        Creates the directory if it doesn't exist.

        Returns:
            Path to .thala/output/ directory
        """
        OUTPUT_DIR.mkdir(exist_ok=True)
        return OUTPUT_DIR

    def generate_timestamp(self) -> str:
        """Generate a timestamp string for output filenames.

        Returns:
            Timestamp in YYYYMMDD_HHMMSS format
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def slugify(self, text: str, max_length: int = 50) -> str:
        """Convert text to a filesystem-safe slug.

        Args:
            text: Text to slugify
            max_length: Maximum length of result

        Returns:
            Slugified string safe for filenames
        """
        return text[:max_length].replace(" ", "_").replace("/", "-").replace("\\", "-")
