"""
Checkpoint management for wrapped research workflow.

Provides file-based checkpointing to support resumption after interruption.
Checkpoints are saved after expensive operations (parallel research, book finding).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default checkpoint directory
CHECKPOINT_DIR = Path.home() / ".thala" / "checkpoints" / "wrapped_research"


def _serialize_datetime(obj: Any) -> Any:
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _deserialize_datetime(data: dict) -> dict:
    """Convert ISO format strings back to datetime objects."""
    datetime_fields = ["started_at", "completed_at"]

    for field in datetime_fields:
        if field in data and data[field]:
            try:
                data[field] = datetime.fromisoformat(data[field])
            except (ValueError, TypeError):
                pass

    # Handle nested workflow results
    for result_field in ["web_result", "academic_result", "book_result"]:
        if data.get(result_field):
            for dt_field in datetime_fields:
                if data[result_field].get(dt_field):
                    try:
                        data[result_field][dt_field] = datetime.fromisoformat(
                            data[result_field][dt_field]
                        )
                    except (ValueError, TypeError):
                        pass

    return data


def get_checkpoint_path(run_id: str) -> Path:
    """Get checkpoint file path for a run.

    Args:
        run_id: The LangSmith run ID (UUID)

    Returns:
        Path to the checkpoint file
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{run_id}.json"


def save_checkpoint(state: dict, phase: str) -> Path:
    """Save workflow state to checkpoint file.

    Called after expensive operations to enable resumption:
    - parallel_research: Web and academic both complete
    - book_query_generated: Theme extracted
    - book_finding: Books complete
    - saved_to_top_of_mind: All records saved

    Args:
        state: Current workflow state dict
        phase: Name of the phase just completed

    Returns:
        Path to the saved checkpoint file
    """
    run_id = state.get("langsmith_run_id", "unknown")
    checkpoint_path = get_checkpoint_path(run_id)

    # Update checkpoint phase tracking
    if "checkpoint_phase" not in state:
        state["checkpoint_phase"] = {
            "parallel_research": False,
            "book_query_generated": False,
            "book_finding": False,
            "saved_to_top_of_mind": False,
        }
    state["checkpoint_phase"][phase] = True
    state["checkpoint_path"] = str(checkpoint_path)

    # Serialize with datetime handling
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(state, f, indent=2, default=_serialize_datetime)

        logger.info(f"Checkpoint saved: {checkpoint_path} (phase: {phase})")
        return checkpoint_path

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise


def load_checkpoint(run_id: str) -> Optional[dict]:
    """Load workflow state from checkpoint.

    Args:
        run_id: The LangSmith run ID (UUID) to load

    Returns:
        State dict if checkpoint exists, None otherwise
    """
    checkpoint_path = get_checkpoint_path(run_id)

    if not checkpoint_path.exists():
        logger.info(f"No checkpoint found for run: {run_id}")
        return None

    try:
        with open(checkpoint_path, "r") as f:
            state = json.load(f)

        # Convert datetime strings back to datetime objects
        state = _deserialize_datetime(state)

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return state

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def get_resume_phase(state: dict) -> str:
    """Determine which phase to resume from based on checkpoint.

    Args:
        state: Loaded checkpoint state

    Returns:
        Name of the node to resume from
    """
    phases = state.get("checkpoint_phase", {})

    if not phases.get("parallel_research"):
        return "parallel_research"
    if not phases.get("book_query_generated"):
        return "generate_book_query"
    if not phases.get("book_finding"):
        return "book_finding"
    if not phases.get("saved_to_top_of_mind"):
        return "save_to_top_of_mind"

    return "generate_final_summary"  # Last phase


def delete_checkpoint(run_id: str) -> bool:
    """Delete checkpoint file after successful completion.

    Args:
        run_id: The LangSmith run ID (UUID)

    Returns:
        True if deleted, False if not found
    """
    checkpoint_path = get_checkpoint_path(run_id)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Checkpoint deleted: {checkpoint_path}")
        return True

    return False
