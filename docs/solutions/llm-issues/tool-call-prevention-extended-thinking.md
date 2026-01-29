---
module: System
date: 2026-01-27
problem_type: llm_issue
component: llm_call
symptoms:
  - "Tool agent makes calls that get ignored when hitting max_tool_calls"
  - "Wasted LLM cost for tool calls whose results are never used"
  - "Empty or corrupted responses when extended thinking is enabled"
  - "AttributeError: 'dict' object has no attribute 'strip' on response"
root_cause: logic_error
resolution_type: code_fix
severity: medium
tags: [tool-agent, extended-thinking, response-parsing, cost-optimization, llm-utils]
---

# Tool Call Prevention and Extended Thinking Response Parsing

## Problem

Two related issues with LLM call configuration edge cases caused wasted API costs and corrupted responses.

### Issue 1: Wasted Tool Calls Near Limits

**Symptom:** Tool agent makes calls that get ignored when hitting `max_tool_calls` limit. The LLM spends tokens planning and executing tool calls, but the results are never processed because the loop terminates immediately after.

```
Tool Loop Behavior (Before Fix):
├── Call 1: search_papers("AI safety")     → Result processed
├── Call 2: get_citations(paper_id)        → Result processed
├── ...
├── Call 11: search_papers("alignment")    → Result processed
└── Call 12: get_abstract(paper_id)        → Result IGNORED (limit hit)
                                              └─ LLM cost wasted
```

**Root Cause:** No warning to the model when approaching limits. The model has no way to know it should wrap up and prepare a final response.

### Issue 2: Extended Thinking Response Parsing

**Symptom:** Empty or corrupted responses when extended thinking is enabled. The `extract_response_content()` function returns thinking content instead of the actual response, or crashes with `AttributeError`.

```python
# Response structure with extended thinking enabled
response.content = [
    {"type": "thinking", "thinking": "Let me analyze this..."},  # ← Code took this
    {"type": "text", "text": "The actual response content"}      # ← Wanted this
]

# Error when calling .strip() on dict
AttributeError: 'dict' object has no attribute 'strip'
```

**Root Cause:** Code assumed the first block in `response.content` list is the text output. With extended thinking, the first block is `{"type": "thinking", ...}` and text is second.

## Solution

### Fix 1: Soft Warning Before Final Tool Iteration

Inject a guidance message when approaching `max_tool_calls` or 90% of `max_total_chars` limit. This encourages the model to complete research and prepare its final response.

```python
# workflows/shared/llm_utils/structured/executors/agent_runner.py

while call_count < max_tool_calls:
    # Check if this is the final allowed iteration
    is_final_iteration = (call_count == max_tool_calls - 1) or (
        total_chars >= max_total_chars * 0.9  # Within 90% of char limit
    )

    if is_final_iteration and call_count > 0:
        working_messages.append({
            "role": "user",
            "content": "You have used most of your available tool calls. "
                      "Please complete your research now and prepare to "
                      "provide your final response. You may make one more "
                      "tool call if essential, but prioritize completion."
        })

    # Get LLM response
    response: AIMessage = await llm_with_tools.ainvoke(working_messages)
    working_messages.append(response)
    # ... rest of loop
```

**Key design decisions:**
- Warning injected at 90% of char limit OR second-to-last call
- Only shows after at least one tool call (`call_count > 0`)
- Phrased as soft guidance, not hard block (allows one more call if essential)
- Message is conversational, not robotic

### Fix 2: Extended Thinking Response Parsing

Updated `extract_response_content()` to find the text block by type, not by position.

```python
# workflows/shared/llm_utils/response_parsing.py

def extract_response_content(response: Any) -> str:
    """Extract text content from various LLM response formats.

    Handles extended thinking responses by finding the text block,
    not just taking the first block (which may be thinking).
    """
    if isinstance(response.content, str):
        return response.content.strip()
    if isinstance(response.content, list) and response.content:
        # Find text block (skip thinking blocks when extended thinking is enabled)
        for block in response.content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    return block.get("text", "").strip()
            elif hasattr(block, "type") and block.type == "text":
                return block.text.strip()
        # Fallback: return first block's text if no explicit text type found
        first_block = response.content[0]
        if isinstance(first_block, dict):
            return first_block.get("text", "").strip()
        if hasattr(first_block, "text"):
            return first_block.text.strip()
        return str(first_block).strip()
    return str(response.content).strip()
```

**Key design decisions:**
- Iterates through blocks looking for `type == "text"`
- Handles both dict format `{"type": "text", "text": "..."}` and object format with `.type` attribute
- Falls back to original behavior if no explicit text type found (backwards compatible)
- Graceful handling of edge cases with `str()` conversion

## Response Format Reference

Extended thinking responses have this structure:

```python
# Standard response (no extended thinking)
response.content = "The response text"
# OR
response.content = [{"type": "text", "text": "The response text"}]

# Extended thinking response
response.content = [
    {
        "type": "thinking",
        "thinking": "Internal reasoning process..."
    },
    {
        "type": "text",
        "text": "The actual response to return"
    }
]
```

## Files Modified

- `workflows/shared/llm_utils/structured/executors/agent_runner.py` - Add soft warning injection
- `workflows/shared/llm_utils/response_parsing.py` - Fix extended thinking parsing
- `workflows/shared/llm_utils/__init__.py` - Export `extract_response_content`

## Prevention

1. **Test with extended thinking enabled** - When working with Opus models, always test with `thinking={"type": "enabled"}` to catch response parsing issues
2. **Don't assume response structure** - Always iterate to find content by type, not position
3. **Provide soft limits to agents** - Inject guidance messages before hard limits to reduce waste
4. **Log tool call metrics** - Track how often agents hit limits to tune thresholds

## Testing Checklist

- [ ] Tool agent completes research gracefully near limits
- [ ] No wasted tool calls after limit warning
- [ ] Extended thinking responses parse correctly
- [ ] Standard responses still work (backwards compatible)
- [ ] Thinking content never leaked to output

## Related Patterns

- [Model Tier Optimization](./model-tier-optimization.md) - Extended thinking only available on Opus
- [Agent Design Patterns](../../patterns/langgraph/agent-design-patterns.md) - Tool agent architecture

## References

- Commit 46785fc: fix(llm-utils): add soft warning before final tool iteration
- Commit f71921e: Fixes extended thinking response parsing in llm_utils
- [Anthropic Extended Thinking Docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
