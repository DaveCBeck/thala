"""
Checkpoint manager for workflow resumption.

Tracks workflow progress at key phases. Each workflow type defines its own
phases - use get_workflow_phases() to retrieve them dynamically.

Uses PID-based process locking to detect crashed processes.

All public methods that perform file I/O are async to avoid blocking the
event loop. The actual I/O is offloaded to a thread pool via asyncio.to_thread().
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .paths import QUEUE_DIR
from .schemas import CurrentWork, WorkflowCheckpoint

logger = logging.getLogger(__name__)

# Default workflow type for backward compatibility
DEFAULT_WORKFLOW_TYPE = "lit_review_full"


def get_workflow_phases(task_type: str) -> list[str]:
    """Get checkpoint phases for a workflow type.

    Args:
        task_type: Workflow type identifier

    Returns:
        Ordered list of phase names
    """
    # Import here to avoid circular imports
    from .workflows import get_phases

    return get_phases(task_type)


class CheckpointManager:
    """Manage workflow checkpoints for resume capability.

    All public methods that perform file I/O are async to avoid blocking
    the event loop. The actual I/O is offloaded to a thread pool via
    asyncio.to_thread().
    """

    def __init__(self, queue_dir: Optional[Path] = None):
        """Initialize the checkpoint manager.

        Args:
            queue_dir: Override queue directory (for testing)
        """
        self.queue_dir = queue_dir or QUEUE_DIR
        self.current_work_file = self.queue_dir / "current_work.json"

        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_orphaned_temps_sync(self) -> int:
        """Synchronous implementation of cleanup_orphaned_temps.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        cleaned = 0
        for tmp_file in self.queue_dir.glob("*.tmp"):
            try:
                tmp_file.unlink()
                logger.info(f"Cleaned up orphaned temp file: {tmp_file.name}")
                cleaned += 1
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {tmp_file}: {e}")
        return cleaned

    async def cleanup_orphaned_temps(self) -> int:
        """Clean up orphaned .tmp files from interrupted writes.

        These files can be left behind if the process is killed during
        an atomic write operation.
        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Returns:
            Number of temp files cleaned up
        """
        return await asyncio.to_thread(self._cleanup_orphaned_temps_sync)

    def _read_current_work_sync(self) -> CurrentWork:
        """Synchronous implementation of _read_current_work.

        This is the actual file I/O logic.
        """
        if self.current_work_file.exists():
            with open(self.current_work_file, "r") as f:
                data = json.load(f)
                # Handle backward compatibility: active_topics -> active_tasks
                if "active_topics" in data and "active_tasks" not in data:
                    data["active_tasks"] = data.pop("active_topics")
                return data
        return {
            "version": "1.0",
            "active_tasks": [],
            "process_locks": {},
        }

    def _write_current_work_sync(self, work: CurrentWork) -> None:
        """Synchronous implementation of _write_current_work.

        This is the actual file I/O logic.
        """
        temp_file = self.current_work_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(work, f, indent=2)
        temp_file.rename(self.current_work_file)

    def _start_work_sync(
        self,
        task_id: str,
        task_type: str,
        langsmith_run_id: str,
    ) -> None:
        """Synchronous implementation of start_work.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self._read_current_work_sync()

        # Get first phase for this workflow type
        phases = get_workflow_phases(task_type)
        initial_phase = phases[0] if phases else "start"

        checkpoint: WorkflowCheckpoint = {
            "task_id": task_id,
            "task_type": task_type,
            "langsmith_run_id": langsmith_run_id,
            "phase": initial_phase,
            "phase_progress": {},
            "phase_outputs": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
            "counters": {},
        }

        # Remove any existing checkpoint for this task
        # Handle both task_id and topic_id for backward compat
        work["active_tasks"] = [
            c for c in work["active_tasks"]
            if c.get("task_id", c.get("topic_id")) != task_id
        ]
        work["active_tasks"].append(checkpoint)
        work["process_locks"][task_id] = str(os.getpid())

        self._write_current_work_sync(work)
        logger.info(f"Started work on task {task_id} ({task_type})")

    async def start_work(
        self,
        task_id: str,
        task_type: str,
        langsmith_run_id: str,
    ) -> None:
        """Record that work has started on a task.

        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task UUID
            task_type: Workflow type (e.g., "lit_review_full", "web_research")
            langsmith_run_id: LangSmith run ID
        """
        await asyncio.to_thread(
            self._start_work_sync,
            task_id,
            task_type,
            langsmith_run_id,
        )

    def _update_checkpoint_sync(
        self,
        task_id: str,
        phase: str,
        phase_outputs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Synchronous implementation of update_checkpoint.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self._read_current_work_sync()

        for checkpoint in work["active_tasks"]:
            cp_task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")
            if cp_task_id == task_id:
                checkpoint["phase"] = phase
                checkpoint["last_checkpoint_at"] = datetime.now(timezone.utc).isoformat()

                # Store phase outputs for resumption
                if phase_outputs is not None:
                    if "phase_outputs" not in checkpoint:
                        checkpoint["phase_outputs"] = {}
                    checkpoint["phase_outputs"].update(phase_outputs)

                # Store all kwargs in counters
                if "counters" not in checkpoint:
                    checkpoint["counters"] = {}

                for key, value in kwargs.items():
                    checkpoint["counters"][key] = value

                break

        self._write_current_work_sync(work)
        logger.debug(f"Checkpoint: {task_id} -> {phase}")

    async def update_checkpoint(
        self,
        task_id: str,
        phase: str,
        phase_outputs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Update checkpoint for a task.

        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task being processed
            phase: Current phase name
            phase_outputs: Outputs from completed phases for resumption
            **kwargs: Additional phase-specific data stored in counters
        """
        await asyncio.to_thread(
            self._update_checkpoint_sync,
            task_id,
            phase,
            phase_outputs,
            **kwargs,
        )

    def _complete_work_sync(self, task_id: str) -> None:
        """Synchronous implementation of complete_work.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self._read_current_work_sync()

        work["active_tasks"] = [
            c for c in work["active_tasks"]
            if c.get("task_id", c.get("topic_id")) != task_id
        ]
        work["process_locks"].pop(task_id, None)

        self._write_current_work_sync(work)
        logger.info(f"Completed work on task {task_id}")

    async def complete_work(self, task_id: str) -> None:
        """Mark work as complete and remove checkpoint.

        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task UUID
        """
        await asyncio.to_thread(self._complete_work_sync, task_id)

    async def fail_work(self, task_id: str) -> None:
        """Remove checkpoint after work failed.

        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task UUID
        """
        # Same as complete_work - just remove the checkpoint
        await self.complete_work(task_id)

    def _get_incomplete_work_sync(self) -> list[WorkflowCheckpoint]:
        """Synchronous implementation of get_incomplete_work.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self._read_current_work_sync()
        incomplete = []

        for checkpoint in work["active_tasks"]:
            task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")
            lock_pid = work["process_locks"].get(task_id)

            # Check if process is still alive
            if lock_pid:
                try:
                    os.kill(int(lock_pid), 0)  # Signal 0 = check existence
                    # Process still alive, skip
                    continue
                except (OSError, ValueError):
                    # Process dead, this is incomplete work
                    pass

            incomplete.append(checkpoint)

        return incomplete

    async def get_incomplete_work(self) -> list[WorkflowCheckpoint]:
        """Get list of incomplete work items.

        Filters out items where the owning process is still alive.
        These are work items that can be resumed.
        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Returns:
            List of checkpoints for incomplete work
        """
        return await asyncio.to_thread(self._get_incomplete_work_sync)

    def _get_active_work_sync(self) -> list[WorkflowCheckpoint]:
        """Synchronous implementation of get_active_work.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self._read_current_work_sync()
        return work["active_tasks"]

    async def get_active_work(self) -> list[WorkflowCheckpoint]:
        """Get all active work items (including those with live processes).

        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Returns:
            List of all active checkpoints
        """
        return await asyncio.to_thread(self._get_active_work_sync)

    def _get_checkpoint_sync(self, task_id: str) -> Optional[WorkflowCheckpoint]:
        """Synchronous implementation of get_checkpoint.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self._read_current_work_sync()

        for checkpoint in work["active_tasks"]:
            cp_task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")
            if cp_task_id == task_id:
                return checkpoint

        return None

    async def get_checkpoint(self, task_id: str) -> Optional[WorkflowCheckpoint]:
        """Get checkpoint for a specific task.

        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task UUID

        Returns:
            Checkpoint if found, None otherwise
        """
        return await asyncio.to_thread(self._get_checkpoint_sync, task_id)

    def can_resume_from_phase(self, checkpoint: WorkflowCheckpoint) -> str:
        """Determine which phase to resume from.

        Some phases can be resumed directly, others need to restart
        from an earlier phase.

        Note: This method does not perform file I/O, so it remains synchronous.

        Args:
            checkpoint: Workflow checkpoint

        Returns:
            Phase name to resume from
        """
        phase = checkpoint["phase"]
        task_type = checkpoint.get("task_type", DEFAULT_WORKFLOW_TYPE)
        phases = get_workflow_phases(task_type)

        # If phase is valid for this workflow, return it
        if phase in phases:
            return phase

        # Otherwise return first phase
        return phases[0] if phases else "start"

    def get_phase_index(self, phase: str, task_type: str = DEFAULT_WORKFLOW_TYPE) -> int:
        """Get index of a phase in the workflow.

        Note: This method does not perform file I/O, so it remains synchronous.

        Args:
            phase: Phase name
            task_type: Workflow type

        Returns:
            Index (0-based), or -1 if not found
        """
        phases = get_workflow_phases(task_type)
        try:
            return phases.index(phase)
        except ValueError:
            return -1

    def is_phase_complete(
        self,
        checkpoint: WorkflowCheckpoint,
        phase: str,
    ) -> bool:
        """Check if a phase has been completed.

        Note: This method does not perform file I/O, so it remains synchronous.

        Args:
            checkpoint: Workflow checkpoint
            phase: Phase to check

        Returns:
            True if the checkpoint is past this phase
        """
        task_type = checkpoint.get("task_type", DEFAULT_WORKFLOW_TYPE)
        current_idx = self.get_phase_index(checkpoint["phase"], task_type)
        target_idx = self.get_phase_index(phase, task_type)

        if current_idx < 0 or target_idx < 0:
            return False

        return current_idx > target_idx

    def get_completed_phases(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> set[str]:
        """Get set of phases that completed before the checkpoint phase.

        Note: This method does not perform file I/O, so it remains synchronous.

        Args:
            checkpoint: Workflow checkpoint

        Returns:
            Set of phase names that are complete
        """
        task_type = checkpoint.get("task_type", DEFAULT_WORKFLOW_TYPE)
        phases = get_workflow_phases(task_type)
        current_phase = checkpoint.get("phase", "")

        try:
            current_idx = phases.index(current_phase)
            return set(phases[:current_idx])
        except ValueError:
            return set()
