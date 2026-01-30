"""
Incremental state manager for mid-phase checkpointing.

Stores partial progress within iterative phases (e.g., paper processing,
supervision loops) to enable resumption from the last checkpoint rather
than restarting the entire phase.

Storage: .thala/queue/incremental/{task_id}.json

Usage:
    mgr = IncrementalStateManager()

    # Save progress every N iterations
    for i, item in enumerate(items):
        results[item.id] = process(item)
        if (i + 1) % 5 == 0:
            mgr.save_progress(task_id, "processing", i + 1, results)

    # On resume, load partial results
    state = mgr.load_progress(task_id, "processing")
    if state:
        results = state["partial_results"]
        start_from = state["iteration_count"]

    # Clear on phase completion
    mgr.clear_progress(task_id)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from typing_extensions import TypedDict

from .paths import QUEUE_DIR

logger = logging.getLogger(__name__)


class IncrementalState(TypedDict):
    """Schema for incremental checkpoint state.

    Stored at .thala/queue/incremental/{task_id}.json
    """

    task_id: str
    phase: str
    iteration_count: int  # Number of items processed
    checkpoint_interval: int  # Every N items (for reference)
    partial_results: dict[str, Any]  # Keyed by identifier (DOI, loop_id, etc.)
    last_checkpoint_at: str  # ISO timestamp


class IncrementalStateManager:
    """Manage incremental checkpoints within workflow phases.

    Provides atomic saves for partial progress, allowing resumption
    from the last checkpoint rather than restarting entire phases.
    """

    def __init__(self, queue_dir: Optional[Path] = None):
        """Initialize the incremental state manager.

        Args:
            queue_dir: Override queue directory (for testing)
        """
        self.queue_dir = queue_dir or QUEUE_DIR
        self.incremental_dir = self.queue_dir / "incremental"
        self.incremental_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_file(self, task_id: str) -> Path:
        """Get path to incremental state file for a task."""
        return self.incremental_dir / f"{task_id}.json"

    def save_progress(
        self,
        task_id: str,
        phase: str,
        iteration_count: int,
        partial_results: dict[str, Any],
        checkpoint_interval: int = 5,
    ) -> None:
        """Save incremental progress for a task.

        Uses atomic write (temp file + rename) to prevent corruption.

        Args:
            task_id: Task being processed
            phase: Current phase name
            iteration_count: Number of items processed
            partial_results: Accumulated results so far (keyed by identifier)
            checkpoint_interval: Interval at which checkpoints are taken
        """
        state: IncrementalState = {
            "task_id": task_id,
            "phase": phase,
            "iteration_count": iteration_count,
            "checkpoint_interval": checkpoint_interval,
            "partial_results": partial_results,
            "last_checkpoint_at": datetime.utcnow().isoformat(),
        }

        state_file = self._get_state_file(task_id)
        temp_file = state_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
            temp_file.rename(state_file)
            logger.info(
                f"Incremental checkpoint: {task_id[:8]} {phase} "
                f"({iteration_count} items, {len(partial_results)} results)"
            )
        except Exception as e:
            # Clean up temp file on failure
            temp_file.unlink(missing_ok=True)
            logger.error(f"Failed to save incremental state: {e}")
            raise

    def load_progress(
        self,
        task_id: str,
        phase: Optional[str] = None,
    ) -> Optional[IncrementalState]:
        """Load incremental progress for a task.

        Args:
            task_id: Task to load progress for
            phase: Optional phase filter - only return if phase matches

        Returns:
            IncrementalState if found (and phase matches), None otherwise
        """
        state_file = self._get_state_file(task_id)

        if not state_file.exists():
            return None

        try:
            with open(state_file, "r") as f:
                state: IncrementalState = json.load(f)

            # Phase filter
            if phase and state.get("phase") != phase:
                logger.debug(
                    f"Incremental state phase mismatch: "
                    f"expected {phase}, got {state.get('phase')}"
                )
                return None

            logger.info(
                f"Loaded incremental state: {task_id[:8]} {state['phase']} "
                f"({state['iteration_count']} items)"
            )
            return state

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted incremental state for {task_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load incremental state: {e}")
            return None

    def clear_progress(self, task_id: str) -> bool:
        """Clear incremental progress for a task.

        Called when a phase completes successfully.

        Args:
            task_id: Task to clear progress for

        Returns:
            True if cleared, False if didn't exist
        """
        state_file = self._get_state_file(task_id)

        if state_file.exists():
            try:
                state_file.unlink()
                logger.debug(f"Cleared incremental state for {task_id[:8]}")
                return True
            except Exception as e:
                logger.warning(f"Failed to clear incremental state: {e}")
                return False

        return False

    def cleanup_orphaned_temps(self) -> int:
        """Clean up orphaned .tmp files from interrupted writes.

        Returns:
            Number of temp files cleaned up
        """
        cleaned = 0
        for tmp_file in self.incremental_dir.glob("*.tmp"):
            try:
                tmp_file.unlink()
                logger.info(f"Cleaned up orphaned temp file: {tmp_file.name}")
                cleaned += 1
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {tmp_file}: {e}")
        return cleaned

    def list_pending(self) -> list[IncrementalState]:
        """List all pending incremental states.

        Useful for debugging and monitoring.

        Returns:
            List of all incremental states
        """
        states = []
        for state_file in self.incremental_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    states.append(json.load(f))
            except Exception:
                continue
        return states
