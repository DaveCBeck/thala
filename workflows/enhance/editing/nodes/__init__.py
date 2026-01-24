"""Node implementations for the editing workflow.

V2 structure nodes handle analyze → rewrite → reassemble.
V1 Enhancement and Polish nodes handle citation enhancement and polishing.
Bridge node connects V2 output to V1 phases.
"""

# V2 Structure Phase nodes
from .v2_analyze import v2_analyze_node
from .v2_rewrite_section import v2_rewrite_section_node
from .v2_reassemble import v2_reassemble_node
from .v2_router import v2_route_to_rewriters, v2_rewrite_router_node

# Bridge node (V2 -> V1)
from .bridge import v2_to_v1_bridge_node

# Citation detection (used in bridge, also exported for routing)
from .detect_citations import (
    extract_citation_keys,
    route_to_enhance_or_polish,
)

# V1 Enhancement Phase nodes
from .enhance_section import (
    route_to_enhance_sections,
    enhance_section_worker,
    assemble_enhancements_node,
)
from .enhance_coherence import enhance_coherence_review_node, route_enhance_iteration

# V1 Polish Phase nodes
from .polish import polish_node

# V1 Finalize nodes
from .finalize import finalize_node

__all__ = [
    # V2 Structure Phase
    "v2_analyze_node",
    "v2_rewrite_router_node",
    "v2_route_to_rewriters",
    "v2_rewrite_section_node",
    "v2_reassemble_node",
    # Bridge
    "v2_to_v1_bridge_node",
    # Citation routing
    "extract_citation_keys",
    "route_to_enhance_or_polish",
    # V1 Enhancement Phase
    "route_to_enhance_sections",
    "enhance_section_worker",
    "assemble_enhancements_node",
    "enhance_coherence_review_node",
    "route_enhance_iteration",
    # V1 Polish Phase
    "polish_node",
    # V1 Finalize
    "finalize_node",
]
