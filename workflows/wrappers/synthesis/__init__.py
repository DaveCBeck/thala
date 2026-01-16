"""Synthesis workflow.

Orchestrates multiple research workflows to create comprehensive
synthesized reports:
1. Academic literature review
2. Supervision (theoretical depth + literature expansion)
3. Web research + book finding (parallel)
4. Synthesis and section writing
5. Editing

Usage:
    from workflows.wrappers.synthesis import synthesis
    result = await synthesis(topic="...", research_questions=[...])
"""

# Direct imports for types (no graph dependencies)
from workflows.wrappers.synthesis.state import SynthesisState, SynthesisInput
from workflows.wrappers.synthesis.quality_presets import (
    SynthesisQualitySettings,
    SYNTHESIS_QUALITY_PRESETS,
)


# Lazy import for the main function to avoid import-time failures
def __getattr__(name: str):
    if name == "synthesis":
        from workflows.wrappers.synthesis.graph.api import synthesis
        return synthesis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "synthesis",
    "SynthesisState",
    "SynthesisInput",
    "SynthesisQualitySettings",
    "SYNTHESIS_QUALITY_PRESETS",
]
