"""Test fixtures for thala testing infrastructure.

This module exports fixtures for:
- Testcontainers (Elasticsearch, ChromaDB)
- Mocks (Zotero, Marker)
- Integrated test infrastructure (test_store_manager)

Usage:
    Import fixtures in conftest.py or directly in test modules.
"""

from .containers import (
    ContainerConfig,
    chroma_container,
    containers,
    es_container,
    es_with_indices,
)
from .mocks import mock_marker, mock_zotero

__all__ = [
    # Container fixtures
    "es_container",
    "chroma_container",
    "containers",
    "es_with_indices",
    "ContainerConfig",
    # Mock fixtures
    "mock_zotero",
    "mock_marker",
]
