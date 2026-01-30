# Logging Configuration

## Overview

Thala uses a centralized logging configuration via `core.config.configure_logging()`. This provides:

- **Two-tier logging**: Console and file outputs with independent log levels
- **Third-party isolation**: Noisy library logs go to a separate file
- **Automatic cleanup**: Keeps 4 most recent log files per type
- **Environment-based config**: No code changes needed to adjust verbosity

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `THALA_LOG_LEVEL_CONSOLE` | Console output level | `WARNING` |
| `THALA_LOG_LEVEL_FILE` | File output level | `INFO` |
| `THALA_LOG_DIR` | Log file directory | `./logs/` |

## Log Files

Stored in `./logs/` (project root):

```
logs/
├── academic_lit_review.log      # Test-specific logs
├── academic_lit_review-3p.log   # Third-party logs for that run
├── thala.log                    # Service logs (default name)
├── thala-3p.log
└── previous/                    # Rotated versions
    ├── thala.1.log
    ├── thala.2.log
    └── ...
```

- **Format**: `{name}.log` and `{name}-3p.log` (stable names)
- **Retention**: 4 previous versions per name (cleanup on startup)
- **Rotation**: Previous logs moved to `previous/` directory

## Third-Party Loggers

These libraries log to the separate `*-3p_*.log` file:

- `httpx`, `httpcore`, `urllib3`
- `asyncio`
- `elasticsearch`
- `chromadb`
- `openai`, `anthropic`
- `hpack`, `charset_normalizer`, `filelock`, `fsspec`, `huggingface_hub`

Note: `langchain`, `langgraph` logs are kept in the main log file for workflow visibility.

## Usage

### Entry Points (services, CLI tools)

Call `configure_logging()` once at startup:

```python
from core.config import configure_logging, configure_langsmith

configure_logging()  # Uses default name "thala"
configure_langsmith()
```

Or with a custom name:

```python
configure_logging("mcp_server")  # Creates mcp_server_datetime.log
```

### All Other Modules

Use the standard pattern (no changes needed):

```python
import logging
logger = logging.getLogger(__name__)

# Then use normally
logger.debug("Cache hit for DOI %s", doi)
logger.info("Starting search for query: %s", query)
logger.warning("Primary search failed, using fallback")
logger.error("Search failed: %s", error)
```

### Test Scripts

Import from `testing.utils` with a descriptive name:

```python
import logging
from testing.utils import configure_logging

configure_logging("academic_lit_review")  # Creates academic_lit_review_datetime.log
logger = logging.getLogger(__name__)
```

## Format

**Console** (compact):
```
WARNING - core.scraping.pdf - PDF extraction failed for https://example.com
```

**File** (detailed):
```
2026-01-13 14:30:52,123 - core.scraping.pdf - WARNING - PDF extraction failed for https://example.com
```

---

## Logging Conventions (for implementation)

### Log Levels

| Level | Use For | Examples |
|-------|---------|----------|
| **DEBUG** | Internal flow, cache operations, verbose detail | `"Cache hit for DOI {doi}"`, `"Parsed {n} results"` |
| **INFO** | Significant operations, workflow milestones | `"Starting search for {query}"`, `"Workflow complete"` |
| **WARNING** | Recoverable issues, fallbacks triggered | `"Primary search failed, using fallback"`, `"Rate limited"` |
| **ERROR** | Failures that affect results | `"Search failed: {e}"`, `"Unable to connect"` |

### Guidelines

**DO:**
- Include relevant context: `logger.info(f"Processing {len(items)} items for '{query}'")`
- Include exceptions in errors: `logger.error(f"Failed to fetch {url}: {e}")`
- Use `logger.exception()` when you need the full traceback

**DON'T:**
- Log sensitive data (API keys, credentials, tokens in URLs)
- Log redundant info (same data in caller and callee)
- Use INFO for routine success paths (use DEBUG instead)
- Use print() for debugging (use logger.debug())

### What to Log at Each Level

**DEBUG** (verbose, for troubleshooting):
- Cache hits/misses
- Local vs cloud fallback decisions
- Intermediate processing steps
- API response summaries

**INFO** (operational visibility):
- Service/workflow start and completion
- Major phase transitions
- Configuration being used
- Summary counts (e.g., "Found 15 papers")

**WARNING** (needs attention but not broken):
- Fallback paths triggered
- Rate limiting encountered
- Missing optional data
- Deprecated usage detected

**ERROR** (something failed):
- API/service failures
- Required data missing
- Unrecoverable errors (with exception details)
