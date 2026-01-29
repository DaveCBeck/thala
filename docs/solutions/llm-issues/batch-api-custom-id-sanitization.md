---
module: batch_processor
date: 2026-01-04
problem_type: api_validation_error
component: BatchProcessor
symptoms:
  - "anthropic.BadRequestError: custom_id must match ^[a-zA-Z0-9_-]{1,64}$"
  - "Batch submission fails with DOI-based custom_ids"
  - "custom_id validation error for identifiers with slashes or dots"
root_cause: invalid_custom_id_format
resolution_type: input_sanitization
severity: medium
tags: [batch-api, custom-id, sanitization, doi, validation]
---

# Batch API Custom ID Sanitization

## Problem

Batch API submissions failed when custom_ids contained invalid characters:

```
anthropic.BadRequestError: Invalid custom_id "extract-10.1234/abc.def.123":
custom_id must match ^[a-zA-Z0-9_-]{1,64}$
```

This occurred when using DOIs as custom_ids, which contain slashes (`/`) and dots (`.`).

## Root Cause

**DOIs contain characters invalid for Anthropic batch API custom_ids.**

The Anthropic batch API requires `custom_id` to match `^[a-zA-Z0-9_-]{1,64}$`:
- Only alphanumeric, underscore, and hyphen allowed
- Maximum 64 characters

DOIs like `10.1234/journal.pone.0001234` contain:
- Slashes (`/`) - invalid
- Dots (`.`) - invalid
- Often exceed 64 characters

The code was passing raw DOIs as custom_ids:

```python
# PROBLEMATIC: DOI contains invalid characters
processor.add_request(
    custom_id=f"extract-{paper['doi']}",  # "extract-10.1234/foo.bar"
    prompt=...,
)
```

## Solution

**Centralize custom_id sanitization in BatchProcessor with automatic ID mapping.**

### Sanitization Function

```python
# workflows/shared/batch_processor.py

import re


def sanitize_custom_id(identifier: str) -> str:
    """Convert identifier to valid Anthropic batch custom_id.

    The API requires custom_id to match pattern ^[a-zA-Z0-9_-]{1,64}$.
    This replaces any invalid character with underscore and truncates to 64 chars.
    """
    # Replace any character that's not alphanumeric, underscore, or hyphen
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
    return sanitized[:64]
```

### Automatic Sanitization in BatchProcessor

```python
class BatchProcessor:
    def __init__(self, ...):
        # ...
        # Map sanitized custom_id back to original for result lookup
        self._id_mapping: dict[str, str] = {}  # sanitized -> original

    def _build_batch_requests(self) -> list[dict]:
        """Convert pending requests to API format.

        Automatically sanitizes custom_ids and stores mapping for result lookup.
        """
        batch_requests = []
        for req in self.pending_requests:
            # Sanitize custom_id and store mapping
            sanitized_id = sanitize_custom_id(req.custom_id)
            self._id_mapping[sanitized_id] = req.custom_id

            batch_requests.append({
                "custom_id": sanitized_id,  # Use sanitized
                "params": {...},
            })
        return batch_requests

    async def _fetch_results(self, results_url: str) -> dict[str, BatchResult]:
        """Fetch and parse batch results.

        Results are keyed by original (unsanitized) custom_id for caller convenience.
        """
        results: dict[str, BatchResult] = {}

        for result_data in parsed_results:
            sanitized_id = result_data["custom_id"]
            # Map back to original ID for caller convenience
            original_id = self._id_mapping.get(sanitized_id, sanitized_id)

            results[original_id] = BatchResult(
                custom_id=original_id,  # Return original ID
                success=...,
                content=...,
            )

        return results
```

### Transparent to Callers

Callers use original IDs; sanitization is handled internally:

```python
from workflows.shared.batch_processor import BatchProcessor

processor = BatchProcessor()

# Caller uses raw DOI as custom_id
processor.add_request(
    custom_id=f"extract-{paper['doi']}",  # "extract-10.1234/foo.bar"
    prompt=...,
)

results = await processor.execute_batch()

# Results keyed by ORIGINAL ID (not sanitized)
result = results.get(f"extract-{paper['doi']}")  # Works!
```

## Sanitization Examples

| Original ID | Sanitized ID |
|-------------|--------------|
| `extract-10.1234/foo.bar` | `extract-10_1234_foo_bar` |
| `relevance-10.1038/s41586-024-07487` | `relevance-10_1038_s41586-024-07487` |
| `a` * 100 | `a` * 64 (truncated) |

## Files Modified

- `workflows/shared/batch_processor.py` - Added `sanitize_custom_id()`, auto-mapping
- `workflows/research/subgraphs/academic_lit_review/paper_processor/extraction.py` - Removed local sanitization (now centralized)

## Prevention

When using custom_ids in batch APIs:

1. **Centralize sanitization**: Handle in the processor, not callers
2. **Map back to original**: Store mapping for transparent result lookup
3. **Test with edge cases**: DOIs, long strings, special characters
4. **Document constraints**: Note API regex requirements

## Testing

```python
def test_sanitize_custom_id():
    """Test custom_id sanitization for Anthropic batch API."""
    # DOI with slashes and dots
    assert sanitize_custom_id("10.1234/foo.bar") == "10_1234_foo_bar"

    # Already valid
    assert sanitize_custom_id("valid-id_123") == "valid-id_123"

    # Truncation
    long_id = "a" * 100
    assert len(sanitize_custom_id(long_id)) == 64

    # Special characters
    assert sanitize_custom_id("id@#$%") == "id____"


async def test_batch_processor_id_mapping():
    """Test that results use original IDs, not sanitized."""
    processor = BatchProcessor()

    original_id = "extract-10.1234/foo.bar"
    processor.add_request(
        custom_id=original_id,
        prompt="test",
        model=ModelTier.HAIKU,
    )

    results = await processor.execute_batch()

    # Should be able to look up by ORIGINAL ID
    assert original_id in results
    assert results[original_id].custom_id == original_id
```

## Related Patterns

- [Batch API Cost Optimization](../../patterns/llm-interaction/batch-api-cost-optimization.md) - BatchProcessor usage
- [Batch API JSON/Structured Output](./batch-api-json-structured-output.md) - Other batch API fixes

## References

- [Anthropic Batch API](https://docs.anthropic.com/en/api/creating-message-batches)
