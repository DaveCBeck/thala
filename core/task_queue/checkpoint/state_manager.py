"""
Checkpoint state management.

Handles the lifecycle of checkpoints: start, update, complete, and fail.
All public methods are async to avoid blocking the event loop.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from ..schemas import WorkflowCheckpoint
from .phase_analyzer import get_workflow_phases
from .storage import CheckpointStorage

logger = logging.getLogger(__name__)


class CheckpointStateManager:
    """Manage checkpoint lifecycle operations."""

    def __init__(self, storage: CheckpointStorage):
        """Initialize state manager.

        Args:
            storage: Checkpoint storage instance
        """
        self.storage = storage

    def _start_work_sync(
        self,
        task_id: str,
        task_type: str,
        langsmith_run_id: str,
    ) -> None:
        """Synchronous implementation of start_work.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self.storage._read_current_work_sync()

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

        self.storage._write_current_work_sync(work)
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
        work = self.storage._read_current_work_sync()

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

        self.storage._write_current_work_sync(work)
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
        work = self.storage._read_current_work_sync()

        work["active_tasks"] = [
            c for c in work["active_tasks"]
            if c.get("task_id", c.get("topic_id")) != task_id
        ]
        work["process_locks"].pop(task_id, None)

        self.storage._write_current_work_sync(work)
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

    def _get_active_work_sync(self) -> list[WorkflowCheckpoint]:
        """Synchronous implementation of get_active_work.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self.storage._read_current_work_sync()
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
        work = self.storage._read_current_work_sync()

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
