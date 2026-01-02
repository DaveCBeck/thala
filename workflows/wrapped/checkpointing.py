"""
Checkpoint management for wrapped research workflow.

Provides file-based checkpointing to support resumption after interruption.
Checkpoints are saved after expensive operations (parallel research, book finding).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from workflows.shared.checkpointing import CheckpointManager, _deserialize_datetime

logger = logging.getLogger(__name__)

# Default checkpoint directory
CHECKPOINT_DIR = Path.home() / ".thala" / "checkpoints" / "wrapped_research"


def _deserialize_wrapped_state(data: dict) -> dict:
    """Deserialize wrapped workflow state with nested datetime fields."""
    # Handle top-level datetime fields
    data = _deserialize_datetime(data)

    # Handle nested workflow results
    for result_field in ["web_result", "academic_result", "book_result"]:
        if data.get(result_field):
            for dt_field in ["started_at", "completed_at"]:
                if data[result_field].get(dt_field):
                    try:
                        data[result_field][dt_field] = datetime.fromisoformat(
                            data[result_field][dt_field]
                        )
                    except (ValueError, TypeError):
                        pass

    return data


# Initialize checkpoint manager
_manager = CheckpointManager(
    checkpoint_dir=str(CHECKPOINT_DIR),
    state_deserializer=_deserialize_wrapped_state,
)


def get_checkpoint_path(run_id: str) -> Path:
    """Get checkpoint file path for a run.

    Args:
        run_id: The LangSmith run ID (UUID)

    Returns:
        Path to the checkpoint file
    """
    return _manager.get_checkpoint_path(run_id)


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

    # Update checkpoint phase tracking
    if "checkpoint_phase" not in state:
        state["checkpoint_phase"] = {
            "parallel_research": False,
            "book_query_generated": False,
            "book_finding": False,
            "saved_to_top_of_mind": False,
        }
    state["checkpoint_phase"][phase] = True
    state["checkpoint_path"] = str(get_checkpoint_path(run_id))

    return _manager.save_checkpoint(run_id, state, phase)


def load_checkpoint(run_id: str) -> Optional[dict]:
    """Load workflow state from checkpoint.

    Args:
        run_id: The LangSmith run ID (UUID) to load

    Returns:
        State dict if checkpoint exists, None otherwise
    """
    return _manager.load_checkpoint(run_id)


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
    return _manager.delete_checkpoint(run_id)
