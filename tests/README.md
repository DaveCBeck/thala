# Testing Infrastructure

This directory contains the test infrastructure for thala, including fixtures, factories, and test suites.

## Quick Start

```bash
# Run unit tests only (fast, no containers)
pytest tests/ -m unit

# Run integration tests (requires Docker for testcontainers)
pytest tests/ -m integration

# Run workflow tests with quality level
pytest tests/integration/workflows/ -m integration --quality quick

# Run all tests with parallel workers
pytest tests/ -n auto --dist loadscope

# Run with verbose output
pytest tests/ -v
```

## Directory Structure

```
tests/
├── README.md           # This file
├── conftest.py         # Pytest configuration and fixture imports
├── test_fixtures.py    # Tests for the test infrastructure itself
├── fixtures/           # Infrastructure-level fixtures
│   ├── __init__.py     # Exports all fixtures
│   ├── containers.py   # Testcontainer fixtures (ES, Chroma)
│   └── mocks.py        # Mock fixtures (Zotero, Marker)
├── factories/          # Test data factories
│   └── __init__.py     # Factory functions for test data
├── utils/              # Test utilities
│   ├── __init__.py     # Exports all utilities
│   ├── quality_analyzer.py   # Quality analysis framework
│   ├── datetime_utils.py     # Duration formatting, ISO parsing
│   ├── file_management.py    # Output file management
│   ├── cli_parser.py         # CLI argument helpers
│   └── result_display.py     # Result formatting
├── unit/               # Unit tests (no external services)
│   ├── core/
│   │   ├── llm_broker/
│   │   ├── scraping/pdf/
│   │   ├── task_queue/
│   │   └── ...
│   └── workflows/
│       └── ...
└── integration/        # Integration tests (uses testcontainers)
    ├── llm_broker/
    ├── llm_utils/
    └── workflows/      # End-to-end workflow tests
        ├── test_academic_lit_review.py
        ├── test_document_processing.py
        ├── test_evening_reads_illustrated.py
        └── test_lit_review_then_enhance.py
```

## Fixture Ownership Policy

| Location | Purpose | Examples |
|----------|---------|----------|
| `tests/fixtures/` | Infrastructure-level fixtures | Testcontainers, transport mocks, StoreManager |
| `tests/factories/` | Domain-specific test data factories | `make_academic_paper()`, `make_zotero_item()` |
| Test modules | Test-specific fixtures | Local setup, parametrized data |

## Available Fixtures

### Container Fixtures (Session-scoped)

These fixtures start real services in Docker containers with dynamic ports:

- **`es_container`**: Elasticsearch testcontainer URL (e.g., `http://localhost:49152`)
- **`chroma_container`**: ChromaDB testcontainer as `(host, port)` tuple
- **`containers`**: Combined `ContainerConfig` with all container info
- **`es_with_indices`**: ES container with thala indices created

### Mock Fixtures (Function-scoped)

These fixtures provide mocks for services that are heavy or require external resources:

- **`mock_zotero`**: AsyncMock with stateful behavior matching `ZoteroStoreProtocol`
- **`mock_marker`**: Monkeypatch on `process_pdf_bytes` returning mock markdown

### Integrated Fixture

- **`test_store_manager`**: Fully configured `StoreManager` with testcontainers + mocks

## Test Markers

Tests should be marked appropriately:

```python
@pytest.mark.unit          # No external services, fast
@pytest.mark.integration   # Uses testcontainers, slower
@pytest.mark.slow          # Takes >30s to run
```

## Factory Functions

Available in `tests.factories`:

```python
from tests.factories import (
    make_zotero_item_create,  # ZoteroItemCreate for add()
    make_zotero_item,         # ZoteroItem as returned by get()
    make_store_record,        # StoreRecord for ES stores
    make_coherence_record,    # CoherenceRecord for coherence store
    make_base_record,         # BaseRecord
    make_academic_paper_content,  # Realistic paper markdown
)

# Create with defaults
item = make_zotero_item_create()

# Override specific fields
book = make_zotero_item_create(
    item_type="book",
    title="Custom Title",
)
```

## Writing Tests

### Unit Tests (No External Services)

```python
import pytest

@pytest.mark.unit
async def test_zotero_mock_add(mock_zotero):
    """Test Zotero mock add operation."""
    from core.stores.zotero.schemas import ZoteroItemCreate

    item = ZoteroItemCreate(itemType="book", fields={"title": "Test"})
    key = await mock_zotero.add(item)

    assert key.startswith("TEST")
```

### Integration Tests (With Testcontainers)

```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
async def test_es_document_roundtrip(test_store_manager):
    """Test ES document storage and retrieval."""
    from tests.factories import make_store_record

    record = make_store_record()
    store = test_store_manager.es_stores.store

    # Add document
    await store.add(record)

    # Retrieve and verify
    doc = await store.get(record.id)
    assert doc is not None
```

## Parallel Test Execution

When running tests with pytest-xdist, use `--dist loadscope` to group tests by module:

```bash
pytest tests/ -n auto --dist loadscope
```

This reduces container count by sharing containers among tests in the same module.

## Testcontainer Requirements

Integration tests require:
- Docker running locally
- Sufficient memory for containers (ES needs ~1GB)
- Network access to pull images (first run)

Container images used:
- `docker.elastic.co/elasticsearch/elasticsearch:9.0.0`
- `chromadb/chroma:latest`

## Troubleshooting

### Containers fail to start

1. Check Docker is running: `docker ps`
2. Check available memory: `docker stats`
3. Try pulling images manually: `docker pull chromadb/chroma:latest`

### Tests hang

1. Check for missing `await` in async code
2. Verify container health checks pass
3. Check for deadlocks in async fixtures

### Port conflicts

Testcontainers use dynamic ports, so conflicts with production services should not occur. If you see port issues:

1. Check no other testcontainers are running
2. Verify Docker networking: `docker network ls`
