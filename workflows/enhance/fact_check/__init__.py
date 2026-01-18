"""Fact-check and reference-check workflow for documents.

This workflow verifies factual claims and citations:
- Pre-screens sections to identify those with verifiable claims
- Fact-checks claims using paper corpus and Perplexity
- Validates citation existence and claim support
- Applies corrections for verified errors

Usage:
    from workflows.enhance.fact_check import fact_check

    result = await fact_check(
        document=my_markdown,
        topic="Machine learning best practices",
        quality="standard",
    )

    verified_doc = result["final_report"]
"""

from .graph import fact_check, create_fact_check_graph, fact_check_graph
from .quality_presets import FACT_CHECK_QUALITY_PRESETS, get_fact_check_quality_settings
from .state import FactCheckState, FactCheckInput, build_initial_state
from .schemas import (
    FactCheckScreeningResult,
    ClaimVerification,
    FactCheckResult,
    CitationValidation,
    ReferenceCheckResult,
    VerifiedEdit,
)

__all__ = [
    # Main API
    "fact_check",
    # Graph
    "create_fact_check_graph",
    "fact_check_graph",
    # Quality
    "FACT_CHECK_QUALITY_PRESETS",
    "get_fact_check_quality_settings",
    # State
    "FactCheckState",
    "FactCheckInput",
    "build_initial_state",
    # Schemas
    "FactCheckScreeningResult",
    "ClaimVerification",
    "FactCheckResult",
    "CitationValidation",
    "ReferenceCheckResult",
    "VerifiedEdit",
]
