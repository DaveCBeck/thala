"""Structural editing workflow for documents.

This workflow improves document structure and coherence through:
- Structural analysis to identify issues
- Section reorganization (moves, merges, consolidation)
- Content generation (introductions, conclusions, transitions)
- Redundancy removal
- Flow polishing

Usage:
    from workflows.enhance.editing import editing

    result = await editing(
        document=my_markdown,
        topic="Machine learning best practices",
        quality="standard",
    )

    edited_doc = result["final_report"]
"""

from .graph import editing, create_editing_graph, editing_graph
from .document_model import DocumentModel, Section, ContentBlock
from .quality_presets import EDITING_QUALITY_PRESETS, get_editing_quality_settings
from .state import EditingState, EditingInput, build_initial_state

__all__ = [
    # Main API
    "editing",
    # Graph
    "create_editing_graph",
    "editing_graph",
    # Document model
    "DocumentModel",
    "Section",
    "ContentBlock",
    # Quality
    "EDITING_QUALITY_PRESETS",
    "get_editing_quality_settings",
    # State
    "EditingState",
    "EditingInput",
    "build_initial_state",
]
