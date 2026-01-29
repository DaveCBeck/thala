---
module: workflows/shared
date: 2026-01-13
problem_type: code_debt
component: llm_utils
symptoms:
  - "Multiple thin LLM wrappers duplicating logic"
  - "No type safety on LLM extraction results"
  - "Manual caching patterns scattered across codebase"
  - "Dead code accumulating from refactors"
root_cause: api_evolution
resolution_type: code_fix
severity: medium
tags: [deprecation, migration, refactoring, type-safety, get_structured_output, pydantic, dead-code]
---

# Systematic Deprecation Migration

## Problem

Over time, the codebase accumulated deprecated functions that were thin wrappers around LLM primitives:

- `summarize_text()` - Basic LLM summarization wrapper
- `extract_json()` / `extract_json_cached()` - Unstructured JSON extraction
- `extract_structured()` - Tool-based structured extraction
- `gather_with_error_collection()` - Batch error handling utility

These functions had several issues:
- **No type safety**: Returned `dict[str, Any]` without validation
- **No retry logic**: Failed silently on JSON parsing errors
- **No strategy selection**: Couldn't leverage batch API or tool-use
- **Scattered caching**: Manual `cache_ttl` parameters at each call site

```python
# ❌ OLD: Untyped, unvalidated extraction
from workflows.shared.llm_utils import extract_json_cached

result = await extract_json_cached(
    text=content,
    system_instructions="Extract metadata...",
    schema_hint=SCHEMA_STRING,  # String, not enforced
    tier=ModelTier.SONNET,
)
title = result.get("title")  # No guarantee this exists
```

## Root Cause

**API evolution without cleanup**: As `get_structured_output()` matured with Pydantic validation, batch API support, and automatic strategy selection, the old wrappers became technical debt but weren't removed.

## Solution

Migrate all callers to modern `get_structured_output()` interface with Pydantic models:

```python
# ✅ NEW: Type-safe extraction with validation
from workflows.shared.llm_utils import get_structured_output, ModelTier
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(default=None)
    authors: list[str] = Field(default_factory=list)
    year: Optional[str] = Field(default=None)

result = await get_structured_output(
    output_schema=DocumentMetadata,
    user_prompt=content,
    system_prompt=SYSTEM_PROMPT,
    tier=ModelTier.SONNET,
)
title = result.title  # Type-safe access
```

### Migration Patterns

#### Pattern 1: extract_json_cached → get_structured_output

**Before:**
```python
metadata = await extract_json_cached(
    text=content,
    system_instructions=METADATA_SYSTEM_PROMPT,
    schema_hint=METADATA_SCHEMA,
    tier=ModelTier.SONNET,
)
```

**After:**
```python
class DocumentMetadata(BaseModel):
    title: Optional[str] = Field(default=None, description="Document title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    year: Optional[str] = Field(default=None, description="Publication year")
    is_multi_author: bool = Field(default=False)

result = await get_structured_output(
    output_schema=DocumentMetadata,
    user_prompt=content,
    system_prompt=DOCUMENT_ANALYSIS_SYSTEM,
    tier=ModelTier.DEEPSEEK_V3,
    enable_prompt_cache=True,
)
metadata = result.model_dump()
```

#### Pattern 2: summarize_text → Direct LLM call

**Before:**
```python
summary = await summarize_text(
    text=findings_text,
    target_words=300,
    context=f"Summary about: {topic}",
    tier=ModelTier.HAIKU,
)
```

**After:**
```python
from workflows.shared.llm_utils import get_llm, ModelTier
from langchain_core.messages import HumanMessage

llm = get_llm(tier=ModelTier.HAIKU)
prompt = f"""Summarize in approximately 300 words about: {topic}

Content:
{findings_text}"""

response = await llm.ainvoke([HumanMessage(content=prompt)])
summary = response.content.strip()
```

#### Pattern 3: gather_with_error_collection → Inline asyncio.gather

**Before:**
```python
from workflows.shared.async_utils import gather_with_error_collection

tasks = [process(doc) for doc in documents]
successes, errors = await gather_with_error_collection(tasks, logger)
# Complex index tracking to recombine
```

**After:**
```python
import asyncio

results = await asyncio.gather(*tasks, return_exceptions=True)

processed = []
for doc, result in zip(documents, results):
    if isinstance(result, Exception):
        logger.warning(f"Failed: {result}")
        processed.append({"status": "failed", "error": str(result)})
    else:
        processed.append(result)
```

### Files Modified

**Migrated callers:**
- `workflows/document_processing/nodes/chapter_detector.py` - extract_structured → get_structured_output
- `workflows/document_processing/nodes/metadata_agent.py` - extract_json_cached → get_structured_output
- `workflows/research/academic_lit_review/paper_processor/extraction/core.py` - extract_json_cached → get_structured_output
- `workflows/research/web_research/nodes/search_memory.py` - summarize_text → direct LLM call
- `workflows/research/web_research/nodes/process_citations/metadata.py` - extract_json → get_structured_output
- `workflows/document_processing/graph.py` - gather_with_error_collection → inline pattern

**Deleted deprecated code:**
- `workflows/shared/llm_utils/text_processors.py` (153 lines)
- `workflows/shared/llm_utils/caching.py` - removed deprecated functions (81 lines)
- `workflows/shared/async_utils.py` - removed gather_with_error_collection (25 lines)
- `workflows/shared/marker_client/client.py` - dead methods (94 lines)
- `workflows/research/academic_lit_review/paper_processor/extraction.py` - duplicate file (692 lines)
- `workflows/shared/token_utils.py` - removed estimate_tokens aliases (13 lines)

**Updated exports:**
- `workflows/shared/__init__.py` - removed deprecated exports, added get_structured_output

## Benefits

- **Type safety**: Pydantic validation catches schema mismatches at runtime
- **Automatic retries**: JSON parsing failures handled transparently
- **Strategy selection**: Automatically chooses tool-use, JSON schema, or batch API
- **Cost savings**: 50% reduction with batch API for bulk extractions
- **Reduced complexity**: 1,645 lines removed, fewer abstractions to understand

## Deprecation Checklist

When deprecating functions in the future:

1. **Create type-safe replacement first** - Pydantic models for all inputs/outputs
2. **Migrate all callers** - Search for usages and update one by one
3. **Update `__init__.py` exports** - Remove deprecated, add new functions
4. **Delete deprecated code** - Don't leave commented code or aliases
5. **Update tests** - Ensure test coverage uses new interfaces

## Prevention

- **Prefer composition over wrappers**: Don't create thin wrappers that add little value
- **Establish type contracts early**: Use Pydantic models from the start
- **Regular cleanup sprints**: Schedule periodic dead code audits
- **Document deprecation in PR**: Show before/after examples in commit message

## Related Patterns

- [LangChain BaseTool to Decorator Migration](../../patterns/llm-interaction/langchain-basetool-to-decorator-migration.md) - Library migration example
- [Monolithic-to-Modular Refactoring](../../patterns/data-pipeline/monolithic-to-modular-refactoring.md) - Re-export pattern for backward compatibility

## References

- [Pydantic Model Config](https://docs.pydantic.dev/latest/concepts/config/)
- [Python asyncio.gather](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
