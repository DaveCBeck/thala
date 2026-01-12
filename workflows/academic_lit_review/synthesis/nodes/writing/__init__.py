"""Writing nodes for synthesis subgraph.

Uses Anthropic Batch API for 50% cost reduction when writing 5+ thematic sections.
"""

from .drafting import (
    write_intro_methodology_node,
    write_discussion_conclusions_node,
)
from .revision import write_thematic_sections_node

__all__ = [
    "write_intro_methodology_node",
    "write_thematic_sections_node",
    "write_discussion_conclusions_node",
]
