"""Synthesis workflow nodes."""

from .lit_review import run_lit_review
from .supervision import run_supervision
from .research_targets import generate_research_targets
from .research_workers import (
    route_to_parallel_research,
    web_research_worker,
    book_finding_worker,
    aggregate_research,
)
from .synthesis import (
    suggest_structure,
    simple_synthesis,
    select_books,
)
from .book_summaries import fetch_book_summaries
from .quality_check import (
    route_to_section_workers,
    write_section_worker,
    check_section_quality,
    assemble_sections,
)
from .editing import run_editing
from .finalize import finalize

__all__ = [
    # Phase 1
    "run_lit_review",
    # Phase 2
    "run_supervision",
    # Phase 3
    "generate_research_targets",
    "route_to_parallel_research",
    "web_research_worker",
    "book_finding_worker",
    "aggregate_research",
    # Phase 4
    "suggest_structure",
    "simple_synthesis",
    "select_books",
    "fetch_book_summaries",
    "route_to_section_workers",
    "write_section_worker",
    "check_section_quality",
    "assemble_sections",
    # Phase 5
    "run_editing",
    # Finalize
    "finalize",
]
