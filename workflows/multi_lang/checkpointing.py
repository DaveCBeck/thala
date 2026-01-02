"""
Checkpointing utilities for multi_lang workflow.

Enables saving and resuming workflow state after expensive operations.
Checkpoints are saved after each language execution to enable resumption.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from workflows.multi_lang.state import MultiLangState, CheckpointPhase


logger = logging.getLogger(__name__)

# Checkpoint directory
CHECKPOINT_DIR = Path.home() / ".thala" / "checkpoints" / "multi_lang"


def _serialize_datetime(obj):
    """Serialize datetime objects for JSON."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _deserialize_datetime(data: dict) -> dict:
    """Deserialize datetime strings back to datetime objects."""
    datetime_fields = ["started_at", "completed_at"]
    for field in datetime_fields:
        if field in data and data[field]:
            try:
                data[field] = datetime.fromisoformat(data[field])
            except (ValueError, TypeError):
                pass

    # Handle nested datetime in language_results
    for result in data.get("language_results", []):
        for field in ["started_at", "completed_at"]:
            if field in result and result[field]:
                try:
                    result[field] = datetime.fromisoformat(result[field])
                except (ValueError, TypeError):
                    pass

    return data


def get_checkpoint_path(run_id: str) -> Path:
    """Get the checkpoint file path for a run."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{run_id}.json"


def save_checkpoint(state: dict, phase: str) -> Optional[Path]:
    """
    Save workflow state after completing a phase.

    Args:
        state: Current workflow state dict
        phase: Phase just completed (e.g., "language_selection", "relevance_checks",
               "language_es", "sonnet_analysis", "opus_integration", "saved_to_store")

    Returns:
        Path to checkpoint file, or None if save failed
    """
    run_id = state.get("langsmith_run_id")
    if not run_id:
        logger.warning("Cannot save checkpoint: no langsmith_run_id in state")
        return None

    checkpoint_path = get_checkpoint_path(run_id)

    try:
        # Update checkpoint phase
        checkpoint_phase = state.get("checkpoint_phase", {}).copy()

        if phase == "language_selection":
            checkpoint_phase["language_selection"] = True
        elif phase == "relevance_checks":
            checkpoint_phase["relevance_checks"] = True
        elif phase.startswith("language_"):
            lang_code = phase.replace("language_", "")
            languages_executed = checkpoint_phase.get("languages_executed", {})
            languages_executed[lang_code] = True
            checkpoint_phase["languages_executed"] = languages_executed
        elif phase == "sonnet_analysis":
            checkpoint_phase["sonnet_analysis"] = True
        elif phase == "opus_integration":
            checkpoint_phase["opus_integration"] = True
        elif phase == "saved_to_store":
            checkpoint_phase["saved_to_store"] = True

        state_to_save = {**state, "checkpoint_phase": checkpoint_phase}

        with open(checkpoint_path, "w") as f:
            json.dump(state_to_save, f, default=_serialize_datetime, indent=2)

        logger.info(f"Checkpoint saved: {phase} -> {checkpoint_path}")
        return checkpoint_path

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return None


def load_checkpoint(run_id: str) -> Optional[dict]:
    """
    Load workflow state from checkpoint.

    Args:
        run_id: The langsmith_run_id of the workflow

    Returns:
        State dict if checkpoint exists, None otherwise
    """
    checkpoint_path = get_checkpoint_path(run_id)

    if not checkpoint_path.exists():
        logger.info(f"No checkpoint found for run {run_id}")
        return None

    try:
        with open(checkpoint_path) as f:
            state = json.load(f)

        state = _deserialize_datetime(state)
        logger.info(f"Checkpoint loaded for run {run_id}")
        return state

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def get_resume_phase(state: dict) -> str:
    """
    Determine which phase to resume from based on checkpoint.

    Returns the name of the node to resume at.
    """
    phase = state.get("checkpoint_phase", {})

    if phase.get("saved_to_store"):
        return "completed"

    if phase.get("opus_integration"):
        return "save_results"

    if phase.get("sonnet_analysis"):
        return "opus_integration"

    # Check if all languages are executed
    languages_executed = phase.get("languages_executed", {})
    target_languages = state.get("languages_with_content") or state.get("target_languages", [])

    if all(languages_executed.get(lang, False) for lang in target_languages):
        return "sonnet_analysis"

    # Find next language to execute
    current_index = state.get("current_language_index", 0)
    if current_index < len(target_languages):
        return "execute_next_language"

    if phase.get("relevance_checks"):
        return "filter_relevant_languages"

    if phase.get("language_selection"):
        mode = state.get("input", {}).get("mode")
        if mode == "all_languages":
            return "check_relevance_batch"
        return "execute_next_language"

    return "select_languages"


def delete_checkpoint(run_id: str) -> bool:
    """
    Delete checkpoint file after successful completion.

    Args:
        run_id: The langsmith_run_id of the workflow

    Returns:
        True if deleted, False otherwise
    """
    checkpoint_path = get_checkpoint_path(run_id)

    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.info(f"Checkpoint deleted for run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False

    return False


def list_checkpoints() -> list[dict]:
    """
    List all available checkpoints.

    Returns:
        List of {run_id, topic, phase, timestamp} dicts
    """
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = []
    for path in CHECKPOINT_DIR.glob("*.json"):
        try:
            with open(path) as f:
                state = json.load(f)

            checkpoints.append({
                "run_id": state.get("langsmith_run_id"),
                "topic": state.get("input", {}).get("topic", "Unknown"),
                "phase": get_resume_phase(state),
                "timestamp": path.stat().st_mtime,
            })
        except Exception:
            continue

    return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
