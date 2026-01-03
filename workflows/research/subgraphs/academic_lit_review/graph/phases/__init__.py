"""Phase node functions for academic literature review workflow."""

from .discovery import discovery_phase_node
from .diffusion import diffusion_phase_node
from .processing import processing_phase_node
from .clustering import clustering_phase_node
from .synthesis import synthesis_phase_node
from .supervision import supervision_phase_node

__all__ = [
    "discovery_phase_node",
    "diffusion_phase_node",
    "processing_phase_node",
    "clustering_phase_node",
    "synthesis_phase_node",
    "supervision_phase_node",
]
