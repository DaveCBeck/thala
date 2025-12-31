"""Supervisor state and completeness calculation."""

from datetime import datetime
from typing_extensions import TypedDict

from .researcher_state import ResearchFinding


class DraftReport(TypedDict):
    """Draft report being refined."""

    content: str
    version: int
    last_updated: datetime
    gaps_remaining: list[str]  # Research gaps to fill


class DiffusionState(TypedDict):
    """State for diffusion algorithm."""

    iteration: int  # Current iteration
    max_iterations: int  # Max before forced completion
    completeness_score: float  # 0-1 estimated completeness
    areas_explored: list[str]  # Topics already researched
    areas_to_explore: list[str]  # Topics still needed
    last_decision: str  # Last decision made by supervisor


def calculate_completeness(
    findings: list[ResearchFinding],
    key_questions: list[str],
    iteration: int,
    max_iterations: int,
    gaps_remaining: list[str] | None = None,
) -> float:
    """Calculate research completeness from multiple signals.

    Uses a weighted multi-signal formula:
    - 40%: Iteration progress (gives baseline progression)
    - 30%: Findings coverage (questions answered with good confidence)
    - 20%: Average confidence of findings
    - 15%: Gap penalty (reduces score based on known gaps)

    This ensures:
    - Score increases during research phase (not stuck at 0%)
    - Score reflects quality (confidence)
    - Natural progression toward 85% threshold for completion

    Args:
        findings: Research findings collected so far
        key_questions: Initial research questions from brief
        iteration: Current iteration number
        max_iterations: Maximum iterations for this depth
        gaps_remaining: Known gaps from draft refinement (optional)

    Returns:
        Completeness score between 0.0 and 1.0
    """
    gaps_remaining = gaps_remaining or []

    # 1. Iteration progress (40% weight) - capped at 90% contribution
    iteration_score = min(iteration / max(max_iterations, 1), 0.9)

    # 2. Findings coverage (30% weight)
    total_questions = max(len(key_questions), 1)
    high_confidence_findings = sum(
        1 for f in findings if f.get("confidence", 0) > 0.5
    )
    coverage_score = min(high_confidence_findings / total_questions, 1.0)

    # 3. Average confidence (20% weight)
    if findings:
        avg_confidence = sum(f.get("confidence", 0.5) for f in findings) / len(findings)
    else:
        avg_confidence = 0.0

    # 4. Gap penalty (15% weight, inverted) - less punishing, capped at 10 gaps
    gap_score = max(0, 1.0 - min(len(gaps_remaining), 10) * 0.05)

    # Weighted sum
    completeness = (
        0.40 * iteration_score
        + 0.30 * coverage_score
        + 0.20 * avg_confidence
        + 0.15 * gap_score
    )

    return min(completeness, 1.0)
