---
module: workflows/document_processing
date: 2026-01-14
problem_type: validation_error
component: chapter_detector
symptoms:
  - "Pydantic ValidationError: Input should be a valid list, got str"
  - "Structured output array fields received as JSON strings"
  - "Chapter detection fails intermittently on large documents"
root_cause: llm_output_format
resolution_type: code_fix
severity: medium
tags: [pydantic, structured-output, json, validation, field-validator, claude, array-fields]
---

# JSON String Edge Case in Structured Output Arrays

## Problem

Claude's structured output sometimes returns array fields as JSON strings instead of proper arrays. This causes Pydantic validation errors when models expect `list[T]` but receive a string.

### Symptoms

```python
# Expected output from Claude:
{
    "headings": [
        {"heading": "Chapter 1", "is_chapter": true},
        {"heading": "Section 1.1", "is_chapter": false}
    ]
}

# Actual output (edge case):
{
    "headings": "[{\"heading\": \"Chapter 1\", \"is_chapter\": true}, ...]"
}
```

```
pydantic_core.ValidationError: 1 validation error for HeadingAnalysisResult
headings
  Input should be a valid list [type=list_type, input_value='[{"heading": "Ch...', input_type=str]
```

### Impact

- Non-deterministic failures in document processing workflows
- More common with larger arrays or complex nested objects
- Difficult to reproduce consistently

## Root Cause

**LLM structured output serialization variance**: Claude's structured output generation sometimes double-encodes array fields, serializing the JSON array into a string representation. This appears to happen:
- With larger arrays (many items)
- With complex nested objects inside arrays
- Non-deterministically based on model variance

The JSON is technically valid but in an unexpected format that breaks Pydantic type coercion.

## Solution

Add a Pydantic field validator with `mode="before"` to parse JSON strings before type coercion:

```python
import json
from typing import Any
from pydantic import BaseModel, Field, field_validator


class HeadingAnalysis(BaseModel):
    """Analysis of a single heading."""
    heading: str = Field(description="Exact heading text")
    is_chapter: bool = Field(description="Whether this is a chapter boundary")
    chapter_author: Optional[str] = Field(default=None)


class HeadingAnalysisResult(BaseModel):
    """Result of heading structure analysis."""
    headings: list[HeadingAnalysis] = Field(description="Analysis of each heading")

    @field_validator("headings", mode="before")
    @classmethod
    def parse_json_string(cls, v: Any) -> list:
        """Handle LLM returning JSON string instead of list.

        This addresses a known issue where Claude's structured output sometimes
        returns arrays as stringified JSON rather than proper array structures.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            raise ValueError(f"headings must be a list, got unparseable string: {v[:100]}...")
        return v if v is not None else []
```

### Key Implementation Details

1. **`mode="before"`**: Runs validation BEFORE Pydantic's standard type coercion
2. **String detection**: Only intercepts when value is a string
3. **JSON parsing**: Deserializes the string representation
4. **Type validation**: Confirms parsed result is actually a list
5. **Error messages**: Includes preview of problematic string for debugging
6. **None handling**: Returns empty list for None values

### Execution Flow

```
Claude returns: {"headings": "[{...}, {...}]"}
                        ↓
    field_validator runs (mode="before")
                        ↓
        Detects value is string
                        ↓
        Calls json.loads() → [{...}, {...}]
                        ↓
    Returns parsed list to Pydantic
                        ↓
    Normal type coercion succeeds
                        ↓
    ✅ HeadingAnalysisResult created
```

### Additional Hardening

Enable stricter JSON schema validation in the API call:

```python
result = await get_structured_output(
    output_schema=HeadingAnalysisResult,
    user_prompt=heading_list,
    system_prompt=system_prompt,
    tier=ModelTier.DEEPSEEK_V3,
    max_tokens=8192,
    use_json_schema_method=True,  # Stricter validation
)
```

## Files Modified

- `workflows/document_processing/nodes/chapter_detector.py` - Added field validators to HeadingAnalysisResult and ChapterClassificationResult

## Prevention

When defining Pydantic models for LLM structured output with array fields:

1. **Always add a field validator** for array fields that parse JSON strings:

```python
@field_validator("my_array_field", mode="before")
@classmethod
def parse_json_string(cls, v: Any) -> list:
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        raise ValueError(f"Expected list, got unparseable string")
    return v if v is not None else []
```

2. **Use `use_json_schema_method=True`** for stricter LLM output validation

3. **Test with large arrays** to catch edge cases during development

## Reusable Pattern

Create a mixin or decorator for common use:

```python
from typing import Any
import json


def json_string_parser(v: Any) -> list:
    """Parse JSON string to list if needed. Use with @field_validator."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        raise ValueError(f"Expected list, got: {v[:100]}...")
    return v if v is not None else []


# Usage:
class MyModel(BaseModel):
    items: list[Item]

    @field_validator("items", mode="before")
    @classmethod
    def parse_items(cls, v: Any) -> list:
        return json_string_parser(v)
```

## Related Solutions

- [Batch API JSON/Structured Output](../llm-issues/batch-api-json-structured-output.md) - Tool-based structured output for batch API
- [Query Generation Extraction Fixes](./query-generation-supervisor-extraction-fixes.md) - Structured output reliability

## Related Patterns

- [Anthropic Claude Extended Thinking](../../patterns/llm-interaction/anthropic-claude-extended-thinking.md) - LLM integration patterns

## References

- [Pydantic Field Validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [Anthropic Structured Output](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
