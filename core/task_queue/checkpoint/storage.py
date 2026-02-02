"""
Checkpoint storage operations.

Handles file I/O, JSON serialization, and cleanup of orphaned temp files.
All public methods are async to avoid blocking the event loop.
"""

import asyncio
import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path

from ..schemas import CurrentWork

logger = logging.getLogger(__name__)


class CheckpointJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable objects in checkpoint data.

    Handles:
    - datetime -> ISO format string
    - date -> ISO format string
    - bytes -> skipped (replaced with placeholder, too large for checkpoints)
    - Path -> string
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, bytes):
            # Skip bytes (e.g., image data) - too large for checkpoints
            # The actual data is saved to files during the workflow
            return f"<bytes: {len(obj)} bytes skipped>"
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


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

        This is the actual file I/O logic. Handles corrupted files gracefully.
        """
        default_work: CurrentWork = {
            "version": "1.0",
            "active_tasks": [],
            "process_locks": {},
        }

        if not self.current_work_file.exists():
            return default_work

        try:
            with open(self.current_work_file, "r") as f:
                data = json.load(f)
                # Handle backward compatibility: active_topics -> active_tasks
                if "active_topics" in data and "active_tasks" not in data:
                    data["active_tasks"] = data.pop("active_topics")
                return data
        except json.JSONDecodeError as e:
            # File corrupted (likely from interrupted write during CTRL-C)
            logger.warning(
                f"Corrupted checkpoint file {self.current_work_file}: {e}. "
                "Starting fresh with no incomplete work."
            )
            # Back up corrupted file for debugging
            backup_path = self.current_work_file.with_suffix(".corrupted")
            try:
                self.current_work_file.rename(backup_path)
                logger.info(f"Corrupted checkpoint backed up to: {backup_path}")
            except OSError as backup_err:
                logger.warning(f"Failed to backup corrupted file: {backup_err}")
            return default_work

    def _write_current_work_sync(self, work: CurrentWork) -> None:
        """Synchronous implementation of _write_current_work.

        Uses unique temp file name to prevent race conditions from concurrent writes.
        Without unique names, overlapping writes can clobber each other's temp files,
        triggering the fallback direct write which is vulnerable to CTRL-C corruption.
        """
        # Unique temp file prevents concurrent writes from interfering
        temp_file = self.current_work_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(work, f, indent=2, cls=CheckpointJSONEncoder)
            temp_file.rename(self.current_work_file)
        except Exception:
            # Clean up temp file on any error
            temp_file.unlink(missing_ok=True)
            raise

    def _archive_current_work_sync(self) -> None:
        """Archive current work file before clearing/overwriting.

        Creates a backup of the current checkpoint state that can be used
        for debugging or recovery if subsequent operations fail.
        """
        if self.current_work_file.exists():
            archive_path = self.current_work_file.with_suffix(".previous.json")
            try:
                import shutil
                shutil.copy2(self.current_work_file, archive_path)
                logger.debug(f"Archived checkpoint to {archive_path}")
            except OSError as e:
                logger.warning(f"Failed to archive checkpoint: {e}")
