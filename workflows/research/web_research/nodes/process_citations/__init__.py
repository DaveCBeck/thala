"""
Process citations node.

Post-processes the final report to:
1. Extract metadata for each cited URL via Translation Server
2. Enhance metadata with LLM using scraped content
3. Create Zotero items for each citation
4. Replace numeric citations [1], [2] with Pandoc-style [@KEY]
"""

from .core import process_citations

__all__ = ["process_citations"]
