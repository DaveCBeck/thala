"""
Workflow state storage for cross-workflow state sharing.

Allows workflows to save their full state keyed by langsmith_run_id,
enabling downstream workflows to retrieve rich state without bloated return types.

This is separate from checkpointing (which is for resumption after interruption).
The state store persists completed workflow state for sharing between workflows.

Usage:
    # In workflow API (at end of workflow)
    from workflows.shared.workflow_state_store import save_workflow_state

    save_workflow_state(
        workflow_name="academic_lit_review",
        run_id=langsmith_run_id,
        state=full_state_dict,
    )

    # In consuming workflow
    from workflows.shared.workflow_state_store import load_workflow_state

    state = load_workflow_state("academic_lit_review", run_id)
    if state:
        paper_corpus = state.get("paper_corpus", {})
"""

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from core.config import is_dev_mode

logger = logging.getLogger(__name__)

# Storage location
STATE_STORE_DIR = Path.home() / ".thala" / "workflow_states"

# Compress states larger than 1MB
COMPRESSION_THRESHOLD = 1_000_000


def _serialize_value(obj: Any) -> Any:
    """JSON serializer for datetime, UUID, and other special types."""
    if isinstance(obj, datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    if isinstance(obj, UUID):
        return {"__type__": "uuid", "value": str(obj)}
    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}
    # Fall back to string representation
    return str(obj)


def _deserialize_hook(obj: dict) -> Any:
    """Object hook for JSON deserialization to restore special types."""
    if isinstance(obj, dict) and "__type__" in obj:
        type_name = obj["__type__"]
        value = obj["value"]
        if type_name == "datetime":
            return datetime.fromisoformat(value)
        if type_name == "uuid":
            return UUID(value)
        if type_name == "path":
            return Path(value)
    return obj


def should_persist_state() -> bool:
    """Check if workflow state should be persisted to disk.

    State is only persisted in dev/test mode to avoid disk I/O in production.
    """
    return is_dev_mode()


def get_state_path(workflow_name: str, run_id: str, compressed: bool = False) -> Path:
    """Get the file path for a workflow state.

    Args:
        workflow_name: Name of the workflow (e.g., "academic_lit_review")
        run_id: The langsmith_run_id (UUID string)
        compressed: Whether to use .json.gz extension

    Returns:
        Path to the state file
    """
    workflow_dir = STATE_STORE_DIR / workflow_name
    workflow_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".json.gz" if compressed else ".json"
    return workflow_dir / f"{run_id}{suffix}"


def save_workflow_state(
    workflow_name: str,
    run_id: str,
    state: dict,
    compress: bool = True,
) -> Optional[Path]:
    """
    Save workflow state to disk keyed by langsmith_run_id.

    Only saves in dev/test mode (when THALA_MODE=dev).

    Args:
        workflow_name: Name of the workflow (e.g., "academic_lit_review")
        run_id: The langsmith_run_id (UUID string)
        state: Complete workflow state dict
        compress: Whether to compress large states (default: True)

    Returns:
        Path to saved state file, or None if persistence is disabled
    """
    if not should_persist_state():
        logger.debug("State persistence disabled (THALA_MODE != dev)")
        return None

    try:
        # Serialize to JSON string first to check size
        state_json = json.dumps(state, default=_serialize_value, indent=2)
        use_compression = compress and len(state_json) > COMPRESSION_THRESHOLD

        path = get_state_path(workflow_name, run_id, compressed=use_compression)

        if use_compression:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(state_json)
            logger.debug(
                f"Saved compressed workflow state ({len(state_json) / 1024:.1f}KB): {path}"
            )
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(state_json)
            logger.debug(
                f"Saved workflow state ({len(state_json) / 1024:.1f}KB): {path}"
            )

        return path

    except Exception as e:
        logger.error(f"Failed to save workflow state: {e}")
        return None


def load_workflow_state(workflow_name: str, run_id: str) -> Optional[dict]:
    """
    Load workflow state from disk by langsmith_run_id.

    Tries compressed format first, then uncompressed.

    Args:
        workflow_name: Name of the workflow
        run_id: The langsmith_run_id (UUID string)

    Returns:
        State dict if found, None otherwise
    """
    # Try compressed first, then uncompressed
    for compressed in [True, False]:
        path = get_state_path(workflow_name, run_id, compressed=compressed)
        if path.exists():
            try:
                if compressed:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        state = json.load(f, object_hook=_deserialize_hook)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        state = json.load(f, object_hook=_deserialize_hook)

                logger.debug(f"Loaded workflow state: {path}")
                return state

            except Exception as e:
                logger.error(f"Failed to load workflow state from {path}: {e}")
                return None

    logger.debug(f"No workflow state found for {workflow_name}/{run_id}")
    return None


def state_exists(workflow_name: str, run_id: str) -> bool:
    """Check if a workflow state exists on disk.

    Args:
        workflow_name: Name of the workflow
        run_id: The langsmith_run_id (UUID string)

    Returns:
        True if state file exists (compressed or uncompressed)
    """
    for compressed in [True, False]:
        if get_state_path(workflow_name, run_id, compressed=compressed).exists():
            return True
    return False


def list_workflow_states(workflow_name: str, limit: int = 100) -> list[dict]:
    """
    List available states for a workflow.

    Args:
        workflow_name: Name of the workflow
        limit: Maximum number of states to return

    Returns:
        List of {run_id, path, timestamp, size_bytes} dicts,
        sorted by modification time (newest first)
    """
    workflow_dir = STATE_STORE_DIR / workflow_name
    if not workflow_dir.exists():
        return []

    states = []
    for path in workflow_dir.glob("*.json*"):
        run_id = path.stem
        # Handle .json.gz by stripping both extensions
        if path.suffix == ".gz":
            run_id = Path(run_id).stem

        try:
            stat = path.stat()
            states.append(
                {
                    "run_id": run_id,
                    "path": str(path),
                    "timestamp": stat.st_mtime,
                    "size_bytes": stat.st_size,
                }
            )
        except OSError:
            continue

    return sorted(states, key=lambda x: x["timestamp"], reverse=True)[:limit]


def cleanup_old_states(workflow_name: str, keep_count: int = 50) -> int:
    """
    Remove old workflow states, keeping only the most recent.

    Args:
        workflow_name: Name of the workflow
        keep_count: Number of recent states to keep

    Returns:
        Number of states deleted
    """
    states = list_workflow_states(workflow_name, limit=10000)
    to_delete = states[keep_count:]

    deleted = 0
    for state_info in to_delete:
        try:
            Path(state_info["path"]).unlink()
            deleted += 1
        except OSError as e:
            logger.debug(f"Failed to delete {state_info['path']}: {e}")

    if deleted > 0:
        logger.debug(f"Cleaned up {deleted} old states for {workflow_name}")

    return deleted


def get_latest_state(workflow_name: str) -> Optional[dict]:
    """
    Get the most recent state for a workflow.

    Convenience function for debugging/inspection.

    Args:
        workflow_name: Name of the workflow

    Returns:
        Most recent state dict, or None if no states exist
    """
    states = list_workflow_states(workflow_name, limit=1)
    if not states:
        return None

    return load_workflow_state(workflow_name, states[0]["run_id"])
