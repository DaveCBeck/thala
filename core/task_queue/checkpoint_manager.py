"""
Checkpoint manager for workflow resumption.

Tracks workflow progress at key phases:
- discovery: Research questions generated
- diffusion: Papers being fetched via citation network
- processing: Papers being processed/summarized
- clustering: Thematic clustering
- synthesis: Draft synthesis complete

Uses PID-based process locking to detect crashed processes.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schemas import CurrentWork, WorkflowCheckpoint

logger = logging.getLogger(__name__)

# Storage location (project root / topic_queue)
QUEUE_DIR = Path(__file__).parent.parent.parent / "topic_queue"
CURRENT_WORK_FILE = QUEUE_DIR / "current_work.json"

# Workflow phases in order
WORKFLOW_PHASES = [
    "discovery",  # Research questions generated, keyword search
    "diffusion",  # Papers being fetched via citation network
    "processing",  # Papers being processed/summarized
    "clustering",  # Thematic clustering
    "synthesis",  # Draft synthesis complete
    "supervision",  # Supervision loops (optional)
    "complete",  # Fully done
]


class CheckpointManager:
    """Manage workflow checkpoints for resume capability."""

    def __init__(self, queue_dir: Optional[Path] = None):
        """Initialize the checkpoint manager.

        Args:
            queue_dir: Override queue directory (for testing)
        """
        self.queue_dir = queue_dir or QUEUE_DIR
        self.current_work_file = self.queue_dir / "current_work.json"

        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def _read_current_work(self) -> CurrentWork:
        """Read current work from disk."""
        if self.current_work_file.exists():
            with open(self.current_work_file, "r") as f:
                return json.load(f)
        return {
            "version": "1.0",
            "active_topics": [],
            "process_locks": {},
        }

    def _write_current_work(self, work: CurrentWork) -> None:
        """Write current work atomically."""
        temp_file = self.current_work_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(work, f, indent=2)
        temp_file.rename(self.current_work_file)

    def start_work(
        self,
        topic_id: str,
        langsmith_run_id: str,
    ) -> None:
        """Record that work has started on a topic.

        Args:
            topic_id: Task UUID
            langsmith_run_id: LangSmith run ID
        """
        work = self._read_current_work()

        checkpoint: WorkflowCheckpoint = {
            "topic_id": topic_id,
            "langsmith_run_id": langsmith_run_id,
            "phase": "discovery",
            "phase_progress": {},
            "started_at": datetime.utcnow().isoformat(),
            "last_checkpoint_at": datetime.utcnow().isoformat(),
            "papers_discovered": 0,
            "papers_processed": 0,
            "diffusion_stage": 0,
            "clusters_generated": False,
            "synthesis_complete": False,
            "supervision_loop": None,
        }

        # Remove any existing checkpoint for this topic
        work["active_topics"] = [
            c for c in work["active_topics"] if c["topic_id"] != topic_id
        ]
        work["active_topics"].append(checkpoint)
        work["process_locks"][topic_id] = str(os.getpid())

        self._write_current_work(work)
        logger.info(f"Started work on topic {topic_id}")

    def update_checkpoint(
        self,
        topic_id: str,
        phase: str,
        **kwargs,
    ) -> None:
        """Update checkpoint for a topic.

        Args:
            topic_id: Task being processed
            phase: Current phase name
            **kwargs: Additional phase-specific data
                (papers_discovered, papers_processed, etc.)
        """
        work = self._read_current_work()

        for checkpoint in work["active_topics"]:
            if checkpoint["topic_id"] == topic_id:
                checkpoint["phase"] = phase
                checkpoint["last_checkpoint_at"] = datetime.utcnow().isoformat()

                # Update known fields directly
                known_fields = {
                    "papers_discovered",
                    "papers_processed",
                    "diffusion_stage",
                    "clusters_generated",
                    "synthesis_complete",
                    "supervision_loop",
                }

                for key, value in kwargs.items():
                    if key in known_fields:
                        checkpoint[key] = value
                    else:
                        checkpoint["phase_progress"][key] = value

                break

        self._write_current_work(work)
        logger.debug(f"Checkpoint: {topic_id} -> {phase}")

    def complete_work(self, topic_id: str) -> None:
        """Mark work as complete and remove checkpoint.

        Args:
            topic_id: Task UUID
        """
        work = self._read_current_work()

        work["active_topics"] = [
            c for c in work["active_topics"] if c["topic_id"] != topic_id
        ]
        work["process_locks"].pop(topic_id, None)

        self._write_current_work(work)
        logger.info(f"Completed work on topic {topic_id}")

    def fail_work(self, topic_id: str) -> None:
        """Remove checkpoint after work failed.

        Args:
            topic_id: Task UUID
        """
        # Same as complete_work - just remove the checkpoint
        self.complete_work(topic_id)

    def get_incomplete_work(self) -> list[WorkflowCheckpoint]:
        """Get list of incomplete work items.

        Filters out items where the owning process is still alive.
        These are work items that can be resumed.

        Returns:
            List of checkpoints for incomplete work
        """
        work = self._read_current_work()
        incomplete = []

        for checkpoint in work["active_topics"]:
            topic_id = checkpoint["topic_id"]
            lock_pid = work["process_locks"].get(topic_id)

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

    def get_active_work(self) -> list[WorkflowCheckpoint]:
        """Get all active work items (including those with live processes).

        Returns:
            List of all active checkpoints
        """
        work = self._read_current_work()
        return work["active_topics"]

    def get_checkpoint(self, topic_id: str) -> Optional[WorkflowCheckpoint]:
        """Get checkpoint for a specific topic.

        Args:
            topic_id: Task UUID

        Returns:
            Checkpoint if found, None otherwise
        """
        work = self._read_current_work()

        for checkpoint in work["active_topics"]:
            if checkpoint["topic_id"] == topic_id:
                return checkpoint

        return None

    def can_resume_from_phase(self, checkpoint: WorkflowCheckpoint) -> str:
        """Determine which phase to resume from.

        Some phases can be resumed directly, others need to restart
        from an earlier phase.

        Args:
            checkpoint: Workflow checkpoint

        Returns:
            Phase name to resume from
        """
        phase = checkpoint["phase"]

        # Discovery phase: restart from beginning
        if phase == "discovery":
            return "discovery"

        # Diffusion: can resume if papers were discovered
        if phase == "diffusion" and checkpoint["papers_discovered"] > 0:
            return "diffusion"

        # Processing: can resume if some papers processed
        if phase == "processing" and checkpoint["papers_processed"] > 0:
            return "processing"

        # Clustering onwards: can resume if papers were processed
        if phase in ("clustering", "synthesis", "supervision"):
            if checkpoint["papers_processed"] > 0:
                return phase

        # Default: restart from discovery
        return "discovery"

    def get_phase_index(self, phase: str) -> int:
        """Get index of a phase in the workflow.

        Args:
            phase: Phase name

        Returns:
            Index (0-based), or -1 if not found
        """
        try:
            return WORKFLOW_PHASES.index(phase)
        except ValueError:
            return -1

    def is_phase_complete(self, checkpoint: WorkflowCheckpoint, phase: str) -> bool:
        """Check if a phase has been completed.

        Args:
            checkpoint: Workflow checkpoint
            phase: Phase to check

        Returns:
            True if the checkpoint is past this phase
        """
        current_idx = self.get_phase_index(checkpoint["phase"])
        target_idx = self.get_phase_index(phase)

        if current_idx < 0 or target_idx < 0:
            return False

        return current_idx > target_idx
