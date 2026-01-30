"""
Phase validation and resumption logic.

Provides phase-aware analysis of checkpoints and workflow state.
"""

import asyncio
import logging
import os

from ..schemas import WorkflowCheckpoint
from .storage import CheckpointStorage

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
    from ..workflows import get_phases

    return get_phases(task_type)


class PhaseAnalyzer:
    """Analyze and validate workflow phases."""

    def __init__(self, storage: CheckpointStorage):
        """Initialize phase analyzer.

        Args:
            storage: Checkpoint storage instance
        """
        self.storage = storage

    def _get_incomplete_work_sync(self) -> list[WorkflowCheckpoint]:
        """Synchronous implementation of get_incomplete_work.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        work = self.storage._read_current_work_sync()
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
