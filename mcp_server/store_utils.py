"""Store access utilities for MCP tools."""

from typing import Any

from .errors import StoreConnectionError


def get_es_stores(stores: dict[str, Any]):
    """Get Elasticsearch stores or raise StoreConnectionError."""
    es_stores = stores.get("es")
    if not es_stores:
        raise StoreConnectionError("elasticsearch", "Elasticsearch stores not initialized")
    return es_stores


def get_es_substore(stores: dict[str, Any], substore_name: str):
    """Get specific ES substore (who_i_was, forgotten, coherence, store)."""
    es_stores = get_es_stores(stores)
    return getattr(es_stores, substore_name)


def get_chroma_store(stores: dict[str, Any]):
    """Get Chroma store or raise StoreConnectionError."""
    chroma = stores.get("chroma")
    if not chroma:
        raise StoreConnectionError("chroma", "Chroma store not initialized")
    return chroma


def get_zotero_store(stores: dict[str, Any]):
    """Get Zotero store or raise StoreConnectionError."""
    zotero = stores.get("zotero")
    if not zotero:
        raise StoreConnectionError("zotero", "Zotero store not initialized")
    return zotero
