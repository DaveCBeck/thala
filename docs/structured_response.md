# LLM Structured Output Guide

All structured output needs should use the unified `get_structured_output()` function, which automatically selects the optimal strategy based on context.

### Import

```python
from workflows.shared.llm_utils import (
    get_structured_output,
    StructuredRequest,
    StructuredOutputConfig,
    ModelTier,
)
```

### Single Request

For a single LLM call that returns a Pydantic model:

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    summary: str = Field(description="Brief summary of findings")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    themes: list[str] = Field(default_factory=list)

result = await get_structured_output(
    output_schema=AnalysisResult,
    user_prompt="Analyze this paper: ...",
    system_prompt="You are an expert analyst...",
    tier=ModelTier.SONNET,
)

# result is a validated AnalysisResult instance
print(result.summary)
print(result.confidence)
```

### Batch Request

For multiple items, pass a list of `StructuredRequest`. Set `prefer_batch_api=True` to use the Anthropic Batch API for 50% cost savings:

```python
requests = [
    StructuredRequest(id="paper-1", user_prompt="Summarize: ..."),
    StructuredRequest(id="paper-2", user_prompt="Summarize: ..."),
    StructuredRequest(id="paper-3", user_prompt="Summarize: ..."),
]

results = await get_structured_output(
    output_schema=PaperSummary,
    requests=requests,
    system_prompt=SUMMARIZER_SYSTEM,
    tier=ModelTier.HAIKU,
)

# results is a BatchResult with a dict of StructuredOutputResult
for paper_id, result in results.results.items():
    if result.success:
        print(f"{paper_id}: {result.value.summary}")
    else:
        print(f"{paper_id} failed: {result.error}")
```

### With Tools (Multi-turn Agent)

When you provide tools, the function automatically uses the tool-agent pattern with a `submit_result` tool for the final output:

```python
from langchain_core.tools import StructuredTool

search_tool = StructuredTool.from_function(
    func=search_papers,
    name="search_papers",
    description="Search for academic papers",
)

result = await get_structured_output(
    output_schema=ResearchOutput,
    user_prompt="Research the topic and provide findings",
    system_prompt=RESEARCHER_SYSTEM,
    tools=[search_tool, fetch_tool],
    tier=ModelTier.SONNET,
    max_tool_calls=10,
)
```

## Configuration Options

### Model Tiers

| Tier | Use Case | Context Window |
|------|----------|----------------|
| `ModelTier.HAIKU` | Quick tasks, classification, simple extraction | 200k |
| `ModelTier.SONNET` | Standard tasks, summarization, analysis | 200k |
| `ModelTier.SONNET_1M` | Long documents requiring extended context | 1M |
| `ModelTier.OPUS` | Complex reasoning, deep analysis | 200k |

### Common Parameters

```python
result = await get_structured_output(
    output_schema=MySchema,
    user_prompt="...",

    # Model configuration
    tier=ModelTier.SONNET,
    max_tokens=4096,
    thinking_budget=8000,  # Extended thinking (Opus recommended)

    # Strategy hints
    use_json_schema_method=True,  # Stricter schema validation
    prefer_batch_api=True,  # Route requests through batch API (50% savings)

    # Reliability
    max_retries=2,
    enable_prompt_cache=True,  # 90% cost savings on cache hits

    # Tool agent options
    tools=[...],
    max_tool_calls=10,
)
```

### Full Configuration Object

For complex scenarios, use `StructuredOutputConfig`:

```python
config = StructuredOutputConfig(
    tier=ModelTier.OPUS,
    max_tokens=8192,
    thinking_budget=8000,
    use_json_schema_method=True,
    prefer_batch_api=False,  # Set True for cost savings (or use THALA_PREFER_BATCH_API env var)
    max_retries=3,
    enable_prompt_cache=True,
    cache_ttl="5m",  # or "1h"
    tools=[...],
    max_tool_calls=15,
    max_tool_result_chars=50000,
)

result = await get_structured_output(
    output_schema=MySchema,
    user_prompt="...",
    config=config,
)
```

## Strategy Selection

The function auto-selects the best strategy:

| Condition | Strategy | Cost |
|-----------|----------|------|
| Tools provided | `TOOL_AGENT` | Standard |
| `prefer_batch_api=True` | `BATCH_TOOL_CALL` | 50% savings |
| Default | `LANGCHAIN_STRUCTURED` | Standard |

**Environment Variable:** Set `THALA_PREFER_BATCH_API=true` to route all requests through the batch API by default. This is useful for development/testing or batch processing pipelines where latency isn't critical.

You can force a strategy:

```python
from workflows.shared.llm_utils import StructuredOutputStrategy

result = await get_structured_output(
    output_schema=MySchema,
    user_prompt="...",
    strategy=StructuredOutputStrategy.JSON_PROMPTING,  # Force fallback
)
```

## Pydantic Schema Best Practices

### Use Descriptive Fields

```python
class PaperAnalysis(BaseModel):
    """Analysis output for an academic paper."""

    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 specific findings from the paper",
    )
    methodology: str = Field(
        default="Not specified",
        description="Brief research method description (1-2 sentences)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis accuracy",
    )
```

### Use Literals for Constrained Choices

```python
from typing import Literal

class Decision(BaseModel):
    action: Literal["approve", "reject", "needs_review"] = Field(
        description="The decision outcome"
    )
    reasoning: str = Field(description="Explanation for the decision")
```

### Use Validators for Complex Logic

```python
from pydantic import field_validator, model_validator

class EditManifest(BaseModel):
    needs_restructuring: bool
    edits: list[Edit] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_edits_required(self) -> 'EditManifest':
        if self.needs_restructuring and not self.edits:
            # Log warning but don't fail - LLM might have valid reason
            pass
        return self
```

## Convenience Functions

### Extract from Text

```python
from workflows.shared.llm_utils import extract_from_text

metadata = await extract_from_text(
    output_schema=PaperMetadata,
    text=paper_content,
    extraction_prompt="Extract paper metadata including title, authors, and abstract",
    tier=ModelTier.HAIKU,
)
```

### Classify Content

```python
from workflows.shared.llm_utils import classify_content

result = await classify_content(
    content=scraped_html,
    classification_schema=ContentType,
    instructions="Classify as: full_text, abstract, or paywall",
    tier=ModelTier.HAIKU,
)
```

## Error Handling

### With Exceptions (Default)

```python
from workflows.shared.llm_utils import StructuredOutputError

try:
    result = await get_structured_output(...)
except StructuredOutputError as e:
    print(f"Failed after {e.attempts} attempts: {e}")
    print(f"Schema: {e.schema.__name__}")
    print(f"Strategy: {e.strategy}")
```

### Without Exceptions

Use `get_structured_output_with_result` for error handling without exceptions:

```python
from workflows.shared.llm_utils import get_structured_output_with_result

result = await get_structured_output_with_result(
    output_schema=MySchema,
    user_prompt="...",
)

if result.success:
    print(result.value)
    print(f"Strategy: {result.strategy_used}")
    print(f"Usage: {result.usage}")
else:
    print(f"Error: {result.error}")
```

## Migration from Legacy Patterns

### From `.with_structured_output()`

**Before:**
```python
from workflows.shared.llm_utils import get_llm, ModelTier

llm = get_llm(ModelTier.SONNET, max_tokens=4096)
structured_llm = llm.with_structured_output(MySchema, method="json_schema")
result = await structured_llm.ainvoke([
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": prompt},
])
```

**After:**
```python
from workflows.shared.llm_utils import get_structured_output, ModelTier

result = await get_structured_output(
    output_schema=MySchema,
    user_prompt=prompt,
    system_prompt=SYSTEM,
    tier=ModelTier.SONNET,
    max_tokens=4096,
    use_json_schema_method=True,
)
```

### From `BatchProcessor`

**Before:**
```python
from workflows.shared.batch_processor import BatchProcessor

processor = BatchProcessor(poll_interval=30)
tool = {
    "name": "extract",
    "input_schema": MySchema.model_json_schema(),
}
for item in items:
    processor.add_request(
        custom_id=item.id,
        prompt=item.prompt,
        model=ModelTier.HAIKU,
        tools=[tool],
        tool_choice={"type": "tool", "name": "extract"},
    )
results = await processor.execute_batch()
# Manual JSON parsing...
```

**After:**
```python
from workflows.shared.llm_utils import get_structured_output, StructuredRequest

requests = [StructuredRequest(id=item.id, user_prompt=item.prompt) for item in items]
results = await get_structured_output(
    output_schema=MySchema,
    requests=requests,
    system_prompt=SYSTEM,
    tier=ModelTier.HAIKU,
)
# Results are already validated Pydantic models
```

### From `extract_json_cached`

**Before:**
```python
from workflows.shared.llm_utils import extract_json_cached

data = await extract_json_cached(
    text=content,
    system_instructions=SYSTEM,
    tier=ModelTier.HAIKU,
)
# data is an untyped dict
```

**After:**
```python
from workflows.shared.llm_utils import get_structured_output

result = await get_structured_output(
    output_schema=MySchema,
    user_prompt=content,
    system_prompt=SYSTEM,
    tier=ModelTier.HAIKU,
    enable_prompt_cache=True,
)
# result is a validated MySchema instance
```

## Cost Optimization Tips

1. **Use `prefer_batch_api=True` when latency isn't critical**: 50% cost reduction on requests
2. **Enable prompt caching**: 90% savings on repeated system prompts
3. **Use HAIKU for simple tasks**: Classification, simple extraction
4. **Reserve OPUS for complex reasoning**: Deep analysis, multi-step logic
5. **Set appropriate max_tokens**: Don't over-provision

### Batch API Configuration

```python
# Option 1: Per-call
result = await get_structured_output(
    output_schema=MySchema,
    user_prompt="...",
    prefer_batch_api=True,  # 50% savings, higher latency
)

# Option 2: Environment variable (affects all calls)
# export THALA_PREFER_BATCH_API=true

# Option 3: Config object
config = StructuredOutputConfig(prefer_batch_api=True)
result = await get_structured_output(
    output_schema=MySchema,
    user_prompt="...",
    config=config,
)
```
