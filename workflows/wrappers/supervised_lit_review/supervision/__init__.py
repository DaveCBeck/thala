"""Supervision loop for iterative review improvement.

This module implements a supervisor-driven iterative loop that analyzes
a completed literature review for theoretical gaps, runs focused expansions
on identified topics, and integrates findings back into the review.
"""

from .graph import loop1_graph, run_loop1_standalone, Loop1State, Loop1Result
from .types import (
    IdentifiedIssue,
    SupervisorDecision,
    SupervisionState,
    SupervisionExpansion,
    MAX_SUPERVISION_DEPTH,
)
from .focused_expansion import run_focused_expansion

__all__ = [
    # Main API - Loop 1 (theoretical depth)
    "loop1_graph",
    "run_loop1_standalone",
    "Loop1State",
    "Loop1Result",
    "run_focused_expansion",
    # Types and schemas
    "IdentifiedIssue",
    "SupervisorDecision",
    "SupervisionState",
    "SupervisionExpansion",
    "MAX_SUPERVISION_DEPTH",
]
