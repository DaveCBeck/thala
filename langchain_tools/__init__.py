"""
LangChain tools for Thala memory system.

Two tools are provided:
- search_memory: Cross-store semantic search
- expand_context: Deep-dive retrieval ("more about that")
"""

from .base import StoreManager, get_store_manager
from .search_memory import (
    SearchMemoryTool,
    SearchMemoryInput,
    SearchMemoryOutput,
    MemorySearchResult,
)
from .expand_context import (
    ExpandContextTool,
    ExpandContextInput,
    ExpandedContext,
)

__all__ = [
    # Store management
    "StoreManager",
    "get_store_manager",
    # search_memory tool
    "SearchMemoryTool",
    "SearchMemoryInput",
    "SearchMemoryOutput",
    "MemorySearchResult",
    # expand_context tool
    "ExpandContextTool",
    "ExpandContextInput",
    "ExpandedContext",
]
