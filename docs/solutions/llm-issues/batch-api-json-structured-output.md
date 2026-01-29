---
module: batch_processor
date: 2026-01-05
problem_type: json_parsing_error
component: extraction, clustering
symptoms:
  - "json.JSONDecodeError: Extra data: line 1 column 1234"
  - "json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
  - "Empty response content from batch API"
  - "Batch extraction returning malformed JSON"
root_cause: prompting_without_structured_output
resolution_type: schema_enforcement
severity: high
tags: [batch-api, json, tool-use, pydantic, structured-output]
---

# Batch API JSON/Structured Output Failures

## Problem

When using the Anthropic Batch API with text-based prompts asking for JSON output, responses frequently failed to parse:

```python
# Symptoms
json.JSONDecodeError: Extra data: line 1 column 1234 (char 1233)
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
ValueError: Empty response content from batch API
```

## Root Cause

**Prompting for JSON via text is unreliable in batch contexts.**

The batch API processes requests asynchronously and doesn't maintain the same behavioral consistency as synchronous calls. Common failure modes:

1. **Extra content**: Model adds explanatory text before/after JSON
2. **Markdown wrapping**: Output wrapped in ```json fences
3. **Empty responses**: Model returns nothing (especially with complex prompts)
4. **Truncation**: Partial JSON when token limit approached

The existing code used fragile string stripping:

```python
# PROBLEMATIC: Manual JSON extraction
content = result.content.strip()
if content.startswith("```"):
    lines = content.split("\n")
    content = "\n".join(lines[1:-1])  # Strip fences

extracted = json.loads(content)  # Often fails
```

## Solution

**Use tool calling with Pydantic schemas to guarantee valid JSON.**

Tool calling forces the model to output structured data matching the schema exactly. The API handles JSON serialization.

### Step 1: Define Pydantic Schema

```python
from pydantic import BaseModel, Field


class PaperSummarySchema(BaseModel):
    """Schema for full-text paper summary extraction."""

    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 specific findings from the paper",
    )
    methodology: str = Field(
        default="Not specified",
        description="Brief research method description",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Stated limitations from the paper",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="3-5 topic tags for clustering",
    )
```

### Step 2: Build Tool Definition

```python
def _build_extraction_tool() -> tuple[list[dict], dict]:
    """Build Anthropic tool definition for structured output."""
    tool = {
        "name": "extract_summary",
        "description": "Extract structured summary from an academic paper",
        "input_schema": PaperSummarySchema.model_json_schema(),
    }
    tool_choice = {"type": "tool", "name": "extract_summary"}
    return [tool], tool_choice
```

### Step 3: Add to Batch Request

```python
from workflows.shared.batch_processor import BatchProcessor

processor = BatchProcessor(poll_interval=30)

# Build tool for structured output
tools, tool_choice = _build_extraction_tool()

for doi, data in paper_data.items():
    processor.add_request(
        custom_id=f"extract-{doi}",
        prompt=user_prompt,
        model=ModelTier.HAIKU,
        max_tokens=2048,
        system=PAPER_SUMMARY_EXTRACTION_SYSTEM,
        tools=tools,                    # Tool definition
        tool_choice=tool_choice,        # Force tool use
    )

results = await processor.execute_batch()
```

### Step 4: Parse Results (Simplified)

```python
for custom_id, result in results.items():
    if result and result.success:
        # Tool use output is already valid JSON
        extracted = json.loads(result.content)
        # No need to strip fences or handle extra text
```

## BatchProcessor Tool Support

The `BatchProcessor` class supports tools and tool_choice parameters:

```python
# workflows/shared/batch_processor/processor.py

def add_request(
    self,
    custom_id: str,
    prompt: str,
    model: ModelTier = ModelTier.SONNET,
    max_tokens: int = 4096,
    system: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    tools: Optional[list[dict]] = None,       # Tool definitions
    tool_choice: Optional[dict] = None,       # Force specific tool
) -> None:
```

The `RequestBuilder` formats these for the Anthropic API:

```python
# workflows/shared/batch_processor/request_builder.py

def build_batch_requests(self, requests: list[BatchRequest]) -> list[dict]:
    for req in requests:
        api_request = {
            "custom_id": sanitized_id,
            "params": {
                "model": model_id,
                "messages": [{"role": "user", "content": req.prompt}],
                "max_tokens": req.max_tokens,
            }
        }
        if req.tools:
            api_request["params"]["tools"] = req.tools
        if req.tool_choice:
            api_request["params"]["tool_choice"] = req.tool_choice
```

The `ResultParser` extracts tool use content:

```python
# workflows/shared/batch_processor/result_parser.py

def _parse_result(self, result_line: dict) -> BatchResult:
    # For tool use, extract the tool input as content
    message = result_line["result"]["message"]
    for block in message.get("content", []):
        if block.get("type") == "tool_use":
            return BatchResult(
                custom_id=custom_id,
                success=True,
                content=json.dumps(block.get("input", {})),
                # ...
            )
```

## Files Modified

- `workflows/shared/batch_processor/processor.py` - Added tools/tool_choice params
- `workflows/shared/batch_processor/request_builder.py` - Format tools for API
- `workflows/shared/batch_processor/result_parser.py` - Extract tool use content
- `workflows/research/subgraphs/academic_lit_review/paper_processor/extraction.py` - Use tool for summary extraction
- `workflows/research/subgraphs/academic_lit_review/paper_processor/nodes.py` - Use tool for metadata extraction
- `workflows/research/subgraphs/academic_lit_review/clustering/analysis.py` - Use tool for cluster analysis

## Prevention

When using the Batch API for structured output:

1. **Always use tool calling** - Never rely on text prompts for JSON
2. **Define Pydantic schemas** - Get type safety and documentation
3. **Use `tool_choice`** - Force the model to use your tool
4. **Test batch path separately** - Batch and sync paths can behave differently

## Testing

```python
# Verify tool use produces valid JSON
async def test_batch_extraction_uses_tools():
    processor = BatchProcessor(poll_interval=10)

    tools, tool_choice = _build_extraction_tool()
    processor.add_request(
        custom_id="test-1",
        prompt="Extract summary from this text: ...",
        model=ModelTier.HAIKU,
        max_tokens=1024,
        tools=tools,
        tool_choice=tool_choice,
    )

    results = await processor.execute_batch()
    result = results.get("test-1")

    assert result.success
    # Should parse without errors
    data = json.loads(result.content)
    # Should match schema
    assert "key_findings" in data
```

## Related Patterns

- [Batch API Cost Optimization Pattern](../../patterns/llm-interaction/batch-api-cost-optimization.md) - Batch API usage
- [Anthropic Claude Integration with Extended Thinking](../../patterns/llm-interaction/anthropic-claude-extended-thinking.md) - Structured output patterns

## References

- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/tool-use)
- [Pydantic model_json_schema](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_json_schema)
