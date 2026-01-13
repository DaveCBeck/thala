"""Content classification subpackage.

Uses direct Anthropic SDK (no LangChain) for classifying scraped content.
"""

from .classifier import classify_content
from .types import ClassificationResult

__all__ = [
    "classify_content",
    "ClassificationResult",
]
