"""
Checkpoint utilities for test scripts.

Provides save/load functionality for workflow state checkpoints,
enabling resumption of expensive workflows from intermediate states.
"""

import json
import logging
from pathlib import Path

from .file_management import get_output_dir

logger = logging.getLogger(__name__)


def get_checkpoint_dir() -> Path:
    """Get/create checkpoint directory.

    Returns:
        Path to testing/test_data/checkpoints directory
    """
    checkpoint_dir = get_output_dir() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    state: dict,
    name: str,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Save workflow state to checkpoint file.

    Args:
        state: Workflow state dictionary
        name: Checkpoint name (without extension)
        checkpoint_dir: Directory for checkpoints (default: testing/test_data/checkpoints)

    Returns:
        Path to saved checkpoint file
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{name}.json"

    with open(checkpoint_file, "w") as f:
        json.dump(state, f, indent=2, default=str)

    logger.info(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(
    name: str,
    checkpoint_dir: Path | None = None,
) -> dict | None:
    """Load workflow state from checkpoint file.

    Args:
        name: Checkpoint name (without extension)
        checkpoint_dir: Directory for checkpoints (default: testing/test_data/checkpoints)

    Returns:
        Loaded state dictionary, or None if checkpoint doesn't exist
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()

    checkpoint_file = checkpoint_dir / f"{name}.json"

    if not checkpoint_file.exists():
        logger.error(f"Checkpoint not found: {checkpoint_file}")
        return None

    with open(checkpoint_file, "r") as f:
        state = json.load(f)

    logger.info(f"Checkpoint loaded: {checkpoint_file}")
    return state


def list_checkpoints(checkpoint_dir: Path | None = None) -> list[str]:
    """List available checkpoint names.

    Args:
        checkpoint_dir: Directory for checkpoints (default: testing/test_data/checkpoints)

    Returns:
        List of checkpoint names (without extension)
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()

    if not checkpoint_dir.exists():
        return []

    return [f.stem for f in checkpoint_dir.glob("*.json")]
