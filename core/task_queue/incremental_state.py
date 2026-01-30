"""
Incremental state manager for mid-phase checkpointing.

Stores partial progress within iterative phases (e.g., paper processing,
supervision loops) to enable resumption from the last checkpoint rather
than restarting the entire phase.

Storage: .thala/queue/incremental/{task_id}.json.gz

Uses gzip compression to reduce checkpoint size (~10-30x compression for JSON).
For supervision loops, stores only delta state (current_review, iteration, new DOIs)
rather than full corpus. On resume, full state is reconstructed from phase_outputs
and ES queries.

Usage:
    mgr = IncrementalStateManager()

    # Save progress every N iterations
    for i, item in enumerate(items):
        results[item.id] = process(item)
        if (i + 1) % 5 == 0:
            await mgr.save_progress(task_id, "processing", i + 1, results)

    # On resume, load partial results
    state = await mgr.load_progress(task_id, "processing")
    if state:
        results = state["partial_results"]
        start_from = state["iteration_count"]

    # Clear on phase completion
    await mgr.clear_progress(task_id)
"""

import asyncio
import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .paths import INCREMENTAL_DIR
from .schemas import IncrementalState

logger = logging.getLogger(__name__)


class IncrementalStateManager:
    """Manage incremental checkpoints within workflow phases.

    Provides atomic saves for partial progress, allowing resumption
    from the last checkpoint rather than restarting entire phases.

    All public methods are async to avoid blocking the event loop during
    file I/O operations. The actual I/O is offloaded to a thread pool
    via asyncio.to_thread().
    """

    def __init__(self, incremental_dir: Optional[Path] = None) -> None:
        """Initialize the incremental state manager.

        Args:
            incremental_dir: Override incremental directory (for testing)
        """
        self.incremental_dir = incremental_dir or INCREMENTAL_DIR
        self.incremental_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_file(self, task_id: str) -> Path:
        """Get path to incremental state file for a task (gzip compressed)."""
        return self.incremental_dir / f"{task_id}.json.gz"

    def _get_legacy_state_file(self, task_id: str) -> Path:
        """Get path to legacy uncompressed state file for migration."""
        return self.incremental_dir / f"{task_id}.json"

    def _save_progress_sync(
        self,
        task_id: str,
        phase: str,
        iteration_count: int,
        partial_results: dict[str, Any],
        checkpoint_interval: int,
    ) -> None:
        """Synchronous implementation of save_progress.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        state: IncrementalState = {
            "task_id": task_id,
            "phase": phase,
            "iteration_count": iteration_count,
            "checkpoint_interval": checkpoint_interval,
            "partial_results": partial_results,
            "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
        }

        state_file = self._get_state_file(task_id)
        temp_file = state_file.with_suffix(".tmp.gz")

        try:
            # Write with gzip compression (~10-30x size reduction for JSON)
            with gzip.open(temp_file, "wt", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            temp_file.rename(state_file)

            # Log checkpoint with file size for monitoring
            size_kb = state_file.stat().st_size / 1024
            logger.info(
                f"Incremental checkpoint: {task_id[:8]} {phase} "
                f"({iteration_count} items, {len(partial_results)} results, {size_kb:.1f}KB)"
            )
        except Exception as e:
            # Clean up temp file on failure
            temp_file.unlink(missing_ok=True)
            logger.error(f"Failed to save incremental state: {e}")
            raise

    async def save_progress(
        self,
        task_id: str,
        phase: str,
        iteration_count: int,
        partial_results: dict[str, Any],
        checkpoint_interval: int = 5,
    ) -> None:
        """Save incremental progress for a task.

        Uses atomic write (temp file + rename) to prevent corruption.
        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task being processed
            phase: Current phase name
            iteration_count: Number of items processed
            partial_results: Accumulated results so far (keyed by identifier)
            checkpoint_interval: Interval at which checkpoints are taken
        """
        await asyncio.to_thread(
            self._save_progress_sync,
            task_id,
            phase,
            iteration_count,
            partial_results,
            checkpoint_interval,
        )

    def _load_progress_sync(
        self,
        task_id: str,
        phase: Optional[str] = None,
    ) -> Optional[IncrementalState]:
        """Synchronous implementation of load_progress.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        state_file = self._get_state_file(task_id)
        legacy_file = self._get_legacy_state_file(task_id)

        # Check for gzip file first, then fall back to legacy uncompressed
        if state_file.exists():
            file_to_read = state_file
            use_gzip = True
        elif legacy_file.exists():
            file_to_read = legacy_file
            use_gzip = False
            logger.info(f"Found legacy uncompressed checkpoint for {task_id[:8]}")
        else:
            return None

        try:
            if use_gzip:
                with gzip.open(file_to_read, "rt", encoding="utf-8") as f:
                    state: IncrementalState = json.load(f)
            else:
                with open(file_to_read, "r") as f:
                    state = json.load(f)

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

    async def load_progress(
        self,
        task_id: str,
        phase: Optional[str] = None,
    ) -> Optional[IncrementalState]:
        """Load incremental progress for a task.

        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task to load progress for
            phase: Optional phase filter - only return if phase matches

        Returns:
            IncrementalState if found (and phase matches), None otherwise
        """
        return await asyncio.to_thread(self._load_progress_sync, task_id, phase)

    def _clear_progress_sync(self, task_id: str) -> bool:
        """Synchronous implementation of clear_progress.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        state_file = self._get_state_file(task_id)
        legacy_file = self._get_legacy_state_file(task_id)
        cleared = False

        # Clear gzip file
        if state_file.exists():
            try:
                state_file.unlink()
                logger.debug(f"Cleared incremental state for {task_id[:8]}")
                cleared = True
            except Exception as e:
                logger.warning(f"Failed to clear incremental state: {e}")

        # Also clear legacy uncompressed file if it exists
        if legacy_file.exists():
            try:
                legacy_file.unlink()
                logger.debug(f"Cleared legacy incremental state for {task_id[:8]}")
                cleared = True
            except Exception as e:
                logger.warning(f"Failed to clear legacy incremental state: {e}")

        return cleared

    async def clear_progress(self, task_id: str) -> bool:
        """Clear incremental progress for a task.

        Called when a phase completes successfully.
        Clears both gzip and legacy uncompressed files if they exist.
        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Args:
            task_id: Task to clear progress for

        Returns:
            True if cleared, False if didn't exist
        """
        return await asyncio.to_thread(self._clear_progress_sync, task_id)

    def _cleanup_orphaned_temps_sync(self) -> int:
        """Synchronous implementation of cleanup_orphaned_temps.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        cleaned = 0
        # Clean up both .tmp and .tmp.gz files
        for pattern in ["*.tmp", "*.tmp.gz"]:
            for tmp_file in self.incremental_dir.glob(pattern):
                try:
                    tmp_file.unlink()
                    logger.info(f"Cleaned up orphaned temp file: {tmp_file.name}")
                    cleaned += 1
                except OSError as e:
                    logger.warning(f"Failed to clean up temp file {tmp_file}: {e}")
        return cleaned

    async def cleanup_orphaned_temps(self) -> int:
        """Clean up orphaned temp files from interrupted writes.

        Cleans up both .tmp (legacy) and .tmp.gz (current) files.
        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Returns:
            Number of temp files cleaned up
        """
        return await asyncio.to_thread(self._cleanup_orphaned_temps_sync)

    def _list_pending_sync(self) -> list[IncrementalState]:
        """Synchronous implementation of list_pending.

        This is the actual file I/O logic, called via asyncio.to_thread().
        """
        states = []
        seen_task_ids = set()

        # Read gzip compressed files first (preferred)
        for state_file in self.incremental_dir.glob("*.json.gz"):
            try:
                with gzip.open(state_file, "rt", encoding="utf-8") as f:
                    state = json.load(f)
                    states.append(state)
                    seen_task_ids.add(state.get("task_id"))
            except Exception:
                continue

        # Also read legacy uncompressed files (skip if already loaded)
        for state_file in self.incremental_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    if state.get("task_id") not in seen_task_ids:
                        states.append(state)
            except Exception:
                continue

        return states

    async def list_pending(self) -> list[IncrementalState]:
        """List all pending incremental states.

        Useful for debugging and monitoring.
        Reads both gzip and legacy uncompressed files.
        File I/O is offloaded to a thread pool to avoid blocking the event loop.

        Returns:
            List of all incremental states
        """
        return await asyncio.to_thread(self._list_pending_sync)
