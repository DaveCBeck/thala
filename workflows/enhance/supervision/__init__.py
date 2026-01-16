"""Enhancement supervision workflow - Loop 1 (theoretical depth) and Loop 2 (literature expansion).

This module extracts the supervision loops from the full lit review workflow
to enhance existing markdown reports.
"""

from workflows.enhance.supervision.api import enhance_report
from workflows.enhance.supervision.types import EnhanceInput, EnhanceResult, EnhanceState

__all__ = [
    "enhance_report",
    "EnhanceInput",
    "EnhanceState",
    "EnhanceResult",
]
