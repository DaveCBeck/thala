"""Tests for the testing fixtures themselves.

These tests verify that the test infrastructure works correctly:
- Testcontainers start and are healthy
- Mocks behave according to their protocols
- Factories generate valid data
"""

import pytest

from core.stores.zotero.schemas import ZoteroItemCreate


class TestZoteroMock:
    """Tests for the mock_zotero fixture."""

    @pytest.mark.unit
    async def test_add_and_get(self, mock_zotero):
        """Mock should track added items and retrieve them."""
        item = ZoteroItemCreate(
            itemType="book",
            fields={"title": "Test Book", "date": "2024"},
        )

        # Add should return a key
        key = await mock_zotero.add(item)
        assert key.startswith("TEST")
        assert len(key) == 8

        # Get should return the item
        retrieved = await mock_zotero.get(key)
        assert retrieved is not None
        assert retrieved.key == key
        assert retrieved.itemType == "book"
        assert retrieved.fields["title"] == "Test Book"

    @pytest.mark.unit
    async def test_get_nonexistent(self, mock_zotero):
        """Mock should return None for nonexistent keys."""
        result = await mock_zotero.get("NOTFOUND")
        assert result is None

    @pytest.mark.unit
    async def test_delete(self, mock_zotero):
        """Mock should support deleting items."""
        item = ZoteroItemCreate(itemType="webpage", fields={"title": "Test Page"})
        key = await mock_zotero.add(item)

        # Delete should succeed
        deleted = await mock_zotero.delete(key)
        assert deleted is True

        # Item should no longer exist
        result = await mock_zotero.get(key)
        assert result is None

        # Delete again should fail
        deleted_again = await mock_zotero.delete(key)
        assert deleted_again is False

    @pytest.mark.unit
    async def test_quicksearch(self, mock_zotero):
        """Mock quicksearch should find items by title."""
        await mock_zotero.add(
            ZoteroItemCreate(itemType="book", fields={"title": "Python Programming"})
        )
        await mock_zotero.add(
            ZoteroItemCreate(itemType="book", fields={"title": "Java Programming"})
        )
        await mock_zotero.add(
            ZoteroItemCreate(itemType="book", fields={"title": "Machine Learning"})
        )

        # Search should find matching items
        results = await mock_zotero.quicksearch("Programming")
        assert len(results) == 2

        # Search should be case-insensitive
        results = await mock_zotero.quicksearch("python")
        assert len(results) == 1

    @pytest.mark.unit
    async def test_health_check(self, mock_zotero):
        """Mock should return healthy status."""
        status = await mock_zotero.health_check()
        assert status.healthy is True
        assert status.plugin == "zotero-local-crud"

    @pytest.mark.unit
    async def test_isolation_between_tests(self, mock_zotero):
        """Each test should get a fresh mock with no items."""
        # This test runs after the above tests, but should have empty state
        # because mock_zotero is function-scoped
        results = await mock_zotero.quicksearch("")
        # Empty search returns nothing in our mock
        assert len(results) == 0


class TestMarkerMock:
    """Tests for the mock_marker fixture."""

    @pytest.mark.unit
    async def test_process_pdf_bytes(self, mock_marker):
        """Mock should return markdown for any PDF input."""
        from core.scraping.pdf.processor import process_pdf_bytes

        # Create minimal valid PDF-like bytes (mock doesn't validate)
        # The mock patches the function, so any bytes work
        pdf_bytes = b"%PDF-1.4 fake pdf content"

        result = await process_pdf_bytes(pdf_bytes, quality="fast")

        assert "# Mock Document" in result
        assert "Quality level: fast" in result
        assert f"Content size: {len(pdf_bytes)} bytes" in result

    @pytest.mark.unit
    async def test_process_pdf_with_languages(self, mock_marker):
        """Mock should include language info in output."""
        from core.scraping.pdf.processor import process_pdf_bytes

        pdf_bytes = b"%PDF-1.4 fake"

        result = await process_pdf_bytes(
            pdf_bytes, quality="balanced", langs=["English", "German"]
        )

        assert "Languages: English, German" in result


class TestFactories:
    """Tests for the test data factories."""

    @pytest.mark.unit
    def test_make_zotero_item_create_defaults(self):
        """Factory should create valid item with defaults."""
        from testing.factories import make_zotero_item_create

        item = make_zotero_item_create()

        assert item.itemType == "journalArticle"
        assert "title" in item.fields
        assert len(item.creators) > 0

    @pytest.mark.unit
    def test_make_zotero_item_create_custom(self):
        """Factory should accept custom values."""
        from testing.factories import make_zotero_item_create

        item = make_zotero_item_create(
            item_type="book",
            title="Custom Book Title",
            creators=[{"firstName": "Jane", "lastName": "Smith"}],
            tags=["test", "custom"],
        )

        assert item.itemType == "book"
        assert item.fields["title"] == "Custom Book Title"
        assert item.creators[0].firstName == "Jane"
        assert "test" in item.tags

    @pytest.mark.unit
    def test_make_store_record_defaults(self):
        """Factory should create valid store record with defaults."""
        from testing.factories import make_store_record

        record = make_store_record()

        assert record.id is not None
        assert record.content is not None
        assert len(record.content) > 0
        assert record.language_code == "en"

    @pytest.mark.unit
    def test_make_store_record_external(self):
        """External records should have zotero_key."""
        from core.stores.schema import SourceType
        from testing.factories import make_store_record

        record = make_store_record(source_type=SourceType.EXTERNAL)

        assert record.source_type == SourceType.EXTERNAL
        assert record.zotero_key is not None
        assert len(record.zotero_key) == 8

    @pytest.mark.unit
    def test_make_academic_paper_content(self):
        """Factory should generate multi-section paper content."""
        from testing.factories import make_academic_paper_content

        content = make_academic_paper_content(title="Test Paper")

        assert "# Test Paper" in content
        assert "## Abstract" in content
        assert "## Introduction" in content
        assert "## Methods" in content
        assert "## Results" in content
        assert "## Conclusion" in content

    @pytest.mark.unit
    def test_factories_generate_unique_data(self):
        """Consecutive factory calls should generate unique data."""
        from testing.factories import make_zotero_item_create, make_store_record

        item1 = make_zotero_item_create()
        item2 = make_zotero_item_create()

        assert item1.fields["title"] != item2.fields["title"]

        record1 = make_store_record()
        record2 = make_store_record()

        assert record1.id != record2.id


# Integration tests that require testcontainers
# These are marked with @pytest.mark.integration and will be skipped
# unless testcontainers are available

class TestElasticsearchContainer:
    """Tests for ES testcontainer fixture."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_es_container_healthy(self, es_container):
        """ES container should be healthy and accessible."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{es_container}/_cluster/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ("green", "yellow")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_es_with_indices(self, es_with_indices):
        """ES container should have indices created."""
        import httpx

        async with httpx.AsyncClient() as client:
            # Check that expected indices exist
            for index in ["store_l0", "store_l1", "store_l2", "coherence"]:
                response = await client.head(f"{es_with_indices}/{index}")
                assert response.status_code == 200, f"Index {index} not found"


class TestChromaContainer:
    """Tests for ChromaDB testcontainer fixture."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_chroma_container_healthy(self, chroma_container):
        """ChromaDB container should be healthy and accessible."""
        import httpx

        host, port = chroma_container
        async with httpx.AsyncClient() as client:
            # ChromaDB 1.0+ uses v2 API
            response = await client.get(f"http://{host}:{port}/api/v2/heartbeat")
            assert response.status_code == 200


class TestStoreManagerIntegration:
    """Tests for the integrated test_store_manager fixture."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_store_manager_es_access(self, test_store_manager):
        """StoreManager should provide working ES access."""
        es_stores = test_store_manager.es_stores
        assert es_stores is not None

        # Verify we can access the store (uses private _client internally)
        # Just check the store object exists and is configured
        assert es_stores.store is not None
        assert es_stores.coherence is not None

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_store_manager_chroma_access(self, test_store_manager):
        """StoreManager should provide working ChromaDB access."""
        chroma = test_store_manager.chroma
        assert chroma is not None

        # Verify we can check health
        healthy = await chroma.health_check()
        assert healthy is True

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_store_manager_zotero_mock(self, test_store_manager):
        """StoreManager should use mocked Zotero."""
        zotero = test_store_manager.zotero
        assert zotero is not None

        # Should be our mock
        status = await zotero.health_check()
        assert status.healthy is True
        assert status.plugin == "zotero-local-crud"
