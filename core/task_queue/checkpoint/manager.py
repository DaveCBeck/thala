"""
Checkpoint manager facade.

Provides a unified interface for checkpoint operations by delegating to
specialized components: storage, state management, and phase analysis.
"""

from pathlib import Path
from typing import Optional

from ..schemas import WorkflowCheckpoint
from .phase_analyzer import PhaseAnalyzer
from .state_manager import CheckpointStateManager
from .storage import CheckpointStorage


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
        from ..paths import QUEUE_DIR

        self.queue_dir = queue_dir or QUEUE_DIR
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.storage = CheckpointStorage(self.queue_dir)
        self.state_manager = CheckpointStateManager(self.storage)
        self.phase_analyzer = PhaseAnalyzer(self.storage)

        # Expose convenience reference for tests
        self.current_work_file = self.storage.current_work_file

    # Delegate storage operations
    async def cleanup_orphaned_temps(self) -> int:
        """Clean up orphaned .tmp files from interrupted writes."""
        return await self.storage.cleanup_orphaned_temps()

    # Delegate state management operations
    async def start_work(
        self,
        task_id: str,
        task_type: str,
        langsmith_run_id: str,
    ) -> None:
        """Record that work has started on a task."""
        await self.state_manager.start_work(task_id, task_type, langsmith_run_id)

    async def update_checkpoint(
        self,
        task_id: str,
        phase: str,
        phase_outputs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Update checkpoint for a task."""
        await self.state_manager.update_checkpoint(
            task_id, phase, phase_outputs, **kwargs
        )

    async def complete_work(self, task_id: str) -> None:
        """Mark work as complete and remove checkpoint."""
        await self.state_manager.complete_work(task_id)

    async def fail_work(self, task_id: str) -> None:
        """Remove checkpoint after work failed."""
        await self.state_manager.fail_work(task_id)

    async def get_active_work(self) -> list[WorkflowCheckpoint]:
        """Get all active work items (including those with live processes)."""
        return await self.state_manager.get_active_work()

    async def get_checkpoint(self, task_id: str) -> Optional[WorkflowCheckpoint]:
        """Get checkpoint for a specific task."""
        return await self.state_manager.get_checkpoint(task_id)

    # Delegate phase analysis operations
    async def get_incomplete_work(self) -> list[WorkflowCheckpoint]:
        """Get list of incomplete work items."""
        return await self.phase_analyzer.get_incomplete_work()

    def can_resume_from_phase(self, checkpoint: WorkflowCheckpoint) -> str:
        """Determine which phase to resume from."""
        return self.phase_analyzer.can_resume_from_phase(checkpoint)

    def get_phase_index(self, phase: str, task_type: str = "lit_review_full") -> int:
        """Get index of a phase in the workflow."""
        return self.phase_analyzer.get_phase_index(phase, task_type)

    def is_phase_complete(
        self,
        checkpoint: WorkflowCheckpoint,
        phase: str,
    ) -> bool:
        """Check if a phase has been completed."""
        return self.phase_analyzer.is_phase_complete(checkpoint, phase)

    def get_completed_phases(
        self,
        checkpoint: WorkflowCheckpoint,
    ) -> set[str]:
        """Get set of phases that completed before the checkpoint phase."""
        return self.phase_analyzer.get_completed_phases(checkpoint)
