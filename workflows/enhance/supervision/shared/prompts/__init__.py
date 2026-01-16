"""Prompts for supervision loops 1 and 2."""

from .loop1_supervision import (
    INTEGRATOR_SYSTEM,
    INTEGRATOR_USER,
    SUPERVISOR_SYSTEM,
    SUPERVISOR_USER,
)
from .loop2_literature import (
    LOOP2_ANALYZER_SYSTEM,
    LOOP2_ANALYZER_USER,
    LOOP2_INTEGRATOR_SYSTEM,
    LOOP2_INTEGRATOR_USER,
)

__all__ = [
    # Loop 1: Supervision
    "SUPERVISOR_SYSTEM",
    "SUPERVISOR_USER",
    "INTEGRATOR_SYSTEM",
    "INTEGRATOR_USER",
    # Loop 2: Literature
    "LOOP2_ANALYZER_SYSTEM",
    "LOOP2_ANALYZER_USER",
    "LOOP2_INTEGRATOR_SYSTEM",
    "LOOP2_INTEGRATOR_USER",
]
