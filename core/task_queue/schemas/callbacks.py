"""Checkpoint callback type aliases for task queue."""

from typing import Callable, Optional

# =============================================================================
# Checkpoint Callback Type Aliases
# =============================================================================
# Two distinct callback signatures exist for different checkpointing contexts:
# - PhaseCheckpointCallback: Used by runner for workflow phase transitions
# - IncrementalCheckpointCallback: Used by supervision loops for iteration checkpoints

PhaseCheckpointCallback = Callable[[str, Optional[dict]], None]
"""Callback for phase-level checkpoints in the task queue runner.

Args:
    phase: Phase name (e.g., "lit_review", "enhance", "loop1_iteration_2")
    phase_outputs: Optional dict of outputs from the completed phase
"""

IncrementalCheckpointCallback = Callable[[int, dict], None]
"""Callback for incremental checkpoints within iterative phases.

Args:
    iteration: Iteration number (1-indexed)
    results: Dict containing partial results at this iteration
"""
