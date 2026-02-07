"""Mock fixtures for services that are heavy or require external resources.

Provides function-scoped mocks for:
- Zotero: AsyncMock with stateful behavior matching ZoteroStoreProtocol
- Marker: Monkeypatch on process_pdf_bytes returning mock markdown

Usage:
    async def test_zotero_operations(mock_zotero):
        key = await mock_zotero.add(ZoteroItemCreate(...))
        item = await mock_zotero.get(key)
        ...
"""

from textwrap import dedent
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.stores.zotero.schemas import (
    ZoteroHealthStatus,
    ZoteroItem,
    ZoteroItemCreate,
    ZoteroSearchCondition,
    ZoteroSearchResult,
)


@pytest.fixture
def mock_zotero() -> AsyncMock:
    """Mock ZoteroStore with realistic responses.

    Function-scoped to ensure each test gets isolated state.
    Conforms to ZoteroStoreProtocol for Liskov Substitution compliance.

    Returns:
        AsyncMock configured with Zotero-like behavior including:
        - add() that generates keys and tracks items
        - get() that retrieves tracked items
        - search() that returns empty list (override in tests as needed)
        - health_check() that returns healthy status
    """
    mock = AsyncMock()

    # Track created items for get() lookups (isolated per test)
    _items: dict[str, ZoteroItemCreate] = {}
    _counter = 0

    async def mock_add(item: ZoteroItemCreate) -> str:
        nonlocal _counter
        _counter += 1
        key = f"TEST{_counter:04d}"
        _items[key] = item
        return key

    async def mock_get(key: str) -> ZoteroItem | None:
        if key not in _items:
            return None
        item_data = _items[key]
        # Extract counter from key for deterministic itemID
        item_id = int(key.replace("TEST", ""))
        return ZoteroItem(
            key=key,
            itemID=item_id,
            itemType=item_data.itemType,
            version=1,
            libraryID=1,
            fields=item_data.fields,
            creators=[c.model_dump() for c in item_data.creators],
            tags=[{"tag": t} if isinstance(t, str) else t.model_dump() for t in item_data.tags],
        )

    async def mock_update(key: str, updates: Any) -> bool:
        if key not in _items:
            return False
        # Apply updates to tracked item
        item = _items[key]
        if updates.fields:
            item.fields.update(updates.fields)
        return True

    async def mock_delete(key: str) -> bool:
        if key not in _items:
            return False
        del _items[key]
        return True

    async def mock_search(
        conditions: list[ZoteroSearchCondition],
        limit: int = 100,
        include_full_data: bool = False,
    ) -> list[ZoteroSearchResult | ZoteroItem]:
        # Return empty list by default, tests can override via mock.search.return_value
        return []

    async def mock_quicksearch(
        query: str,
        limit: int = 100,
    ) -> list[ZoteroSearchResult]:
        # Simple implementation: search titles in tracked items
        results = []
        for key, item in _items.items():
            title = item.fields.get("title", "")
            if query.lower() in title.lower():
                item_id = int(key.replace("TEST", ""))
                results.append(
                    ZoteroSearchResult(
                        key=key,
                        itemID=item_id,
                        itemType=item.itemType,
                        title=title,
                    )
                )
                if len(results) >= limit:
                    break
        return results

    mock.add.side_effect = mock_add
    mock.get.side_effect = mock_get
    mock.update.side_effect = mock_update
    mock.delete.side_effect = mock_delete
    mock.search.side_effect = mock_search
    mock.quicksearch.side_effect = mock_quicksearch
    mock.health_check.return_value = ZoteroHealthStatus(
        healthy=True,
        status="ok",
        plugin="zotero-local-crud",
        version="1.0.0",
    )
    mock.close.return_value = None

    # Expose internal state for test assertions
    mock._items = _items

    return mock


@pytest.fixture
def mock_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock Marker PDF processing.

    Patches:
    - core.scraping.pdf.processor.process_pdf_bytes: Returns mock markdown
    - core.scraping.pdf.processor.check_marker_available: Returns True
    - core.scraping.pdf.router._marker_available: Set to True (cached state)

    This ensures tests don't make real HTTP calls to Marker and don't
    fall into CPU-only degraded mode.

    The mock returns realistic markdown structure based on input parameters.
    """

    async def mock_process_pdf_bytes(
        content: bytes,
        quality: str = "balanced",
        langs: list[str] | None = None,
        timeout: float | None = None,
        filename: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Mock PDF processing that returns realistic markdown."""
        langs_str = ", ".join(langs) if langs else "English"
        return dedent(f"""\
            # Mock Document

            ## Extracted Content

            This is mock content from Marker service.

            **Processing details:**
            - Quality level: {quality}
            - Languages: {langs_str}
            - Content size: {len(content)} bytes
            - Filename: {filename or "unnamed.pdf"}

            ## Abstract

            This document contains mock extracted text that simulates
            the output of the Marker PDF extraction service. It includes
            typical academic paper structure.

            ## Introduction

            Lorem ipsum dolor sit amet, consectetur adipiscing elit.
            Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

            ## Methods

            The methodology section describes the approach taken.

            ## Results

            Key findings are presented in this section.

            ## Conclusion

            The conclusion summarizes the main points.

            ## References

            1. Mock reference [1]
            2. Mock reference [2]
            3. Mock reference [3]
        """)

    async def mock_check_marker_available() -> bool:
        """Mock health check - always returns True in tests."""
        return True

    # Patch the processor functions
    monkeypatch.setattr(
        "core.scraping.pdf.processor.process_pdf_bytes",
        mock_process_pdf_bytes,
    )
    monkeypatch.setattr(
        "core.scraping.pdf.processor.check_marker_available",
        mock_check_marker_available,
    )

    # Set the cached router state to True (Marker available)
    # This prevents the session check from making HTTP calls
    import core.scraping.pdf.router as router_module

    monkeypatch.setattr(router_module, "_marker_available", True)
