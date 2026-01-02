"""
Multi-language research workflow.

Runs research across multiple languages, producing:
1. A synthesized document integrating all findings
2. A comparative document analyzing cross-language patterns
"""

from workflows.multi_lang.graph.api import multi_lang_research, MultiLangResult
from workflows.multi_lang.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    get_resume_phase,
    list_checkpoints,
)

__all__ = [
    "multi_lang_research",
    "MultiLangResult",
    "save_checkpoint",
    "load_checkpoint",
    "get_resume_phase",
    "list_checkpoints",
]
