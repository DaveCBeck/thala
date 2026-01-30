"""
Checkpoint storage operations.

Handles file I/O, JSON serialization, and cleanup of orphaned temp files.
All public methods are async to avoid blocking the event loop.
"""

import asyncio
import json
import logging
from pathlib import Path

from ..schemas import CurrentWork

logger = logging.getLogger(__name__)


class CheckpointStorage:
    """Handle checkpoint file I/O operations."""

    def __init__(self, queue_dir: Path):
        """Initialize checkpoint storage.

        Args:
            queue_dir: Directory for checkpoint files
        """
        self.queue_dir = queue_dir
        self.current_work_file = self.queue_dir / "current_work.json"

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
        try:
            temp_file.rename(self.current_work_file)
        except FileNotFoundError:
            # Temp file may have been deleted by concurrent cleanup
            logger.warning(f"Temp file {temp_file} disappeared before rename - retrying write")
            with open(self.current_work_file, "w") as f:
                json.dump(work, f, indent=2)
