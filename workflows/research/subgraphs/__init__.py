"""Subgraph implementations for deep research workflow."""

from workflows.research.subgraphs.web_researcher import web_researcher_subgraph
from workflows.research.subgraphs.academic_researcher import academic_researcher_subgraph
from workflows.research.subgraphs.book_researcher import book_researcher_subgraph

__all__ = [
    "web_researcher_subgraph",
    "academic_researcher_subgraph",
    "book_researcher_subgraph",
]
