"""Supervision loop implementations.

This module contains supervision loop types:
- Loop 2: Literature Base Expansion - Identify and integrate missing literature bases
- Loop 3: Structure and Cohesion - Structural editing with two-agent pattern
- Loop 4: Section-level deep editing - Parallel section editing with holistic review
- Loop 4.5: Cohesion Check - Post-editing cohesion validation
- Loop 5: Fact and Reference Checking - Sequential fact/reference verification
"""

from .loop2 import create_loop2_graph, run_loop2_standalone, Loop2State, Loop2Result
from .loop3 import create_loop3_graph, run_loop3_standalone, Loop3State, Loop3Result
from .loop4_editing import (
    create_loop4_graph,
    run_loop4_standalone,
    Loop4State,
    Loop4Result,
)
from .loop4_5_cohesion import check_cohesion, run_loop4_5_standalone
from .loop5_factcheck import create_loop5_graph, run_loop5_standalone, Loop5State

__all__ = [
    # Loop 2
    "create_loop2_graph",
    "run_loop2_standalone",
    "Loop2State",
    "Loop2Result",
    # Loop 3
    "create_loop3_graph",
    "run_loop3_standalone",
    "Loop3State",
    "Loop3Result",
    # Loop 4
    "create_loop4_graph",
    "run_loop4_standalone",
    "Loop4State",
    "Loop4Result",
    # Loop 4.5
    "check_cohesion",
    "run_loop4_5_standalone",
    # Loop 5
    "create_loop5_graph",
    "run_loop5_standalone",
    "Loop5State",
]
