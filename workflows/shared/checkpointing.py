"""
Shared checkpointing utilities for workflows.

Provides file-based checkpointing to support resumption after interruption.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


def _serialize_datetime(obj: Any) -> Any:
    """JSON serializer for datetime and UUID objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _deserialize_datetime(data: dict) -> dict:
    """Convert ISO format strings back to datetime objects in common fields."""
    datetime_fields = ["started_at", "completed_at"]

    for field in datetime_fields:
        if field in data and data[field]:
            try:
                data[field] = datetime.fromisoformat(data[field])
            except (ValueError, TypeError):
                pass

    return data


class CheckpointManager:
    """Manages workflow checkpoints with customizable serialization."""

    def __init__(
        self,
        checkpoint_dir: str,
        state_serializer: Optional[Callable[[dict], dict]] = None,
        state_deserializer: Optional[Callable[[dict], dict]] = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            state_serializer: Optional function to preprocess state before JSON serialization
            state_deserializer: Optional function to postprocess state after JSON deserialization
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_serializer = state_serializer or (lambda x: x)
        self.state_deserializer = state_deserializer or _deserialize_datetime

    def get_checkpoint_path(self, run_id: str) -> Path:
        """Get checkpoint file path for a run."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / f"{run_id}.json"

    def save_checkpoint(self, run_id: str, state: dict, phase: str) -> Path:
        """
        Save workflow state to checkpoint file.

        Args:
            run_id: The run identifier (e.g., LangSmith run ID)
            state: Current workflow state dict
            phase: Name of the phase just completed

        Returns:
            Path to the saved checkpoint file
        """
        checkpoint_path = self.get_checkpoint_path(run_id)

        # Apply custom serialization if provided
        state_to_save = self.state_serializer(state)

        try:
            with open(checkpoint_path, "w") as f:
                json.dump(state_to_save, f, indent=2, default=_serialize_datetime)

            logger.info(f"Checkpoint saved: {checkpoint_path} (phase: {phase})")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, run_id: str) -> Optional[dict]:
        """
        Load workflow state from checkpoint.

        Args:
            run_id: The run identifier to load

        Returns:
            State dict if checkpoint exists, None otherwise
        """
        checkpoint_path = self.get_checkpoint_path(run_id)

        if not checkpoint_path.exists():
            logger.info(f"No checkpoint found for run: {run_id}")
            return None

        try:
            with open(checkpoint_path, "r") as f:
                state = json.load(f)

            # Apply custom deserialization
            state = self.state_deserializer(state)

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self, run_id: str) -> bool:
        """
        Delete checkpoint file after successful completion.

        Args:
            run_id: The run identifier

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self.get_checkpoint_path(run_id)

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info(f"Checkpoint deleted: {checkpoint_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
                return False

        return False

    def get_resume_phase(self, state: dict) -> Optional[str]:
        """
        Determine which phase to resume from based on checkpoint.

        This is a generic implementation. Override in subclasses for workflow-specific logic.

        Args:
            state: Loaded checkpoint state

        Returns:
            Name of the phase/node to resume from, or None
        """
        return state.get("checkpoint_phase")

    def list_checkpoints(self) -> list[dict]:
        """
        List all available checkpoints in this manager's directory.

        Returns:
            List of checkpoint metadata dicts
        """
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for path in self.checkpoint_dir.glob("*.json"):
            try:
                with open(path) as f:
                    state = json.load(f)

                checkpoints.append({
                    "run_id": state.get("langsmith_run_id") or path.stem,
                    "path": str(path),
                    "timestamp": path.stat().st_mtime,
                })
            except Exception:
                continue

        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
