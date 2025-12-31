"""
Async Elasticsearch wrapper for structured storage.

Uses elasticsearch[async] 8.17.0 with native async support.
Routes to different ES instances based on index.

Index structure:
- ES Coherence (9201): store_l0, store_l1, store_l2, coherence
- ES Forgotten (9200): who_i_was, forgotten
"""

from .base import BaseElasticsearchStore
from .client import ElasticsearchStores
from .stores import CoherenceStore, ForgottenStore, MainStore, WhoIWasStore

__all__ = [
    "ElasticsearchStores",
    "BaseElasticsearchStore",
    "CoherenceStore",
    "MainStore",
    "WhoIWasStore",
    "ForgottenStore",
]
