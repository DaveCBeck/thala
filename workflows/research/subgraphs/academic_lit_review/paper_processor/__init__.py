"""Paper processor subgraph for academic literature review.

Processes papers from diffusion through:
1. Full-text acquisition via retrieve-academic service
2. PDF→Markdown processing via document_processing workflow
3. Structured PaperSummary extraction for clustering

Flow:
    START -> acquire_papers -> process_documents -> extract_summaries -> END
"""

from .api import run_paper_processing
from .graph import create_paper_processing_subgraph, paper_processing_subgraph
from .types import PaperProcessingState

__all__ = [
    "PaperProcessingState",
    "paper_processing_subgraph",
    "create_paper_processing_subgraph",
    "run_paper_processing",
]
