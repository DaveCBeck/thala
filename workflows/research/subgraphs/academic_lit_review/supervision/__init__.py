"""Supervision loop for iterative review improvement.

This module implements a supervisor-driven iterative loop that analyzes
a completed literature review for theoretical gaps, runs focused expansions
on identified topics, and integrates findings back into the review.
"""

from .graph import run_supervision, supervision_subgraph
from .types import (
    IdentifiedIssue,
    SupervisorDecision,
    SupervisionState,
    SupervisionExpansion,
    MAX_SUPERVISION_DEPTH,
)
from .focused_expansion import run_focused_expansion

__all__ = [
    # Main API
    "run_supervision",
    "supervision_subgraph",
    "run_focused_expansion",

    # Types and schemas
    "IdentifiedIssue",
    "SupervisorDecision",
    "SupervisionState",
    "SupervisionExpansion",
    "MAX_SUPERVISION_DEPTH",
]
