---
title: "Research Query Generation and Supervisor Extraction Fixes"
module: workflows/research
date: 2025-12-22
problem_type: llm_output
component: research_workflow
symptoms:
  - "Research queries contained system metadata like 'Iteration 3/8, 0% completeness'"
  - "Supervisor-generated research questions included workflow state information"
  - "JSON parsing of LLM responses was unreliable and brittle"
  - "GraphRecursionError on comprehensive research depth"
root_cause: "Prompt contamination allowing operational metadata to bleed into LLM outputs, combined with brittle JSON regex extraction without semantic validation"
resolution_type: code_fix
severity: high
tags: [prompt-contamination, structured-output, pydantic, query-validation, research-workflow]
---

# Research Query Generation and Supervisor Extraction Fixes

## Problem

During comprehensive research testing, the research workflow was generating malformed queries that contained system metadata instead of research-relevant search terms.

### Symptoms

1. **Metadata contamination in queries**: Research queries like `"Iteration 3/8, 0% completeness"` appeared instead of topic-relevant searches
2. **Supervisor questions corrupted**: Supervisor-generated research questions included workflow state information
3. **Unreliable parsing**: JSON extraction from LLM responses frequently failed with various formats
4. **Recursion errors**: `GraphRecursionError` on comprehensive research depth due to unbounded iterations

### Example of Corrupted Output

```python
# Expected queries
["machine learning optimization techniques", "gradient descent algorithms"]

# Actual queries (corrupted)
["Iteration 3/8, 0% completeness", "Research iteration status", "machine learning"]
```

## Root Cause

Two related issues combined to cause this problem:

### 1. Prompt Contamination

The supervisor prompt embedded operational state without clear boundaries:

```python
# BEFORE (problematic)
prompt = f"""
Current Status:
- Iteration: {iteration}/{max_iterations}
- Completeness: {completeness_score}%
- Findings: {len(findings)} sources

Generate research questions...
"""
```

The LLM interpreted the operational metadata as relevant content and echoed it back in the research questions.

### 2. Unreliable JSON Parsing

Query extraction used fragile regex patterns:

```python
# BEFORE (brittle)
questions_match = re.search(r'"questions"\s*:\s*\[(.*?)\]', content, re.DOTALL)
if questions_match:
    questions_text = "[" + questions_match.group(1) + "]"
    questions_raw = json.loads(questions_text)
```

This had no validation that extracted strings were actual research questions.

## Solution

### Step 1: Add Structured Output Models

Created Pydantic models for type-safe LLM responses:

```python
# workflows/research/state.py

class SearchQueries(BaseModel):
    """Structured output for query generation."""
    queries: list[str] = Field(
        description="Search queries relevant to the research topic only"
    )


class QueryValidation(BaseModel):
    """Validation result for a single query."""
    is_relevant: bool
    reason: str


class QueryValidationBatch(BaseModel):
    """Batch validation of queries."""
    validations: list[QueryValidation]


class SupervisorDecision(BaseModel):
    """Structured output for supervisor decisions."""
    action: str = Field(
        description="One of: conduct_research, refine_draft, research_complete, check_fact"
    )
    reasoning: str
    research_questions: list[str] = Field(default_factory=list)
    draft_updates: Optional[str] = None
    remaining_gaps: list[str] = Field(default_factory=list)
```

### Step 2: Use Structured Output for Query Generation

Replaced JSON parsing with `with_structured_output()`:

```python
# workflows/research/subgraphs/researcher.py

async def generate_queries(state: ResearcherState) -> dict[str, Any]:
    """Generate search queries using structured output with validation."""
    llm = get_llm(ModelTier.HAIKU)
    structured_llm = llm.with_structured_output(SearchQueries)

    prompt = f"""Generate 2-3 search queries to research this question:

Question: {question['question']}

Make queries specific and likely to find authoritative sources.
Focus only on the research topic - do not include any system metadata.
"""

    result: SearchQueries = await structured_llm.ainvoke([{"role": "user", "content": prompt}])

    # Validate queries are relevant (defense in depth)
    valid_queries = await validate_queries(
        queries=result.queries,
        research_question=question['question'],
    )

    return {"search_queries": valid_queries}
```

### Step 3: Add LLM-Based Query Validation

Added a validation layer to filter contaminated queries:

```python
# workflows/research/subgraphs/researcher.py

async def validate_queries(
    queries: list[str],
    research_question: str,
) -> list[str]:
    """Validate queries using LLM to ensure they're relevant to the research."""
    llm = get_llm(ModelTier.HAIKU)
    structured_llm = llm.with_structured_output(QueryValidationBatch)

    prompt = f"""Validate whether these search queries are relevant to the research task.

Research Question: {research_question}

Proposed Search Queries:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(queries))}

Reject queries that:
- Contain system metadata (iteration counts, percentages, internal state)
- Are about completely unrelated topics
- Are too vague or generic to be useful
"""

    try:
        result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])

        valid_queries = []
        for query, validation in zip(queries, result.validations):
            if validation.is_relevant:
                valid_queries.append(query)
            else:
                logger.warning(f"Query rejected: {query[:50]}... Reason: {validation.reason}")

        return valid_queries

    except Exception as e:
        logger.warning(f"Query validation failed: {e}, accepting all queries")
        return queries  # Fail open
```

### Step 4: Implement Structured Supervisor Decision

Added structured output for supervisor with fallback:

```python
# workflows/research/nodes/supervisor.py

async def _get_supervisor_decision_structured(
    llm, system_prompt: str, user_prompt: str, brief: dict
) -> tuple[str, dict]:
    """Try to get supervisor decision using structured output."""
    structured_llm = llm.with_structured_output(SupervisorDecision)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    decision: SupervisorDecision = await structured_llm.ainvoke(messages)

    action = decision.action
    action_data = {}

    if action == "conduct_research":
        action_data["questions"] = [
            {"question": q, "context": brief.get("topic", "")}
            for q in decision.research_questions
        ]
    elif action == "refine_draft":
        action_data["updates"] = decision.draft_updates or ""
        action_data["gaps"] = decision.remaining_gaps

    return action, action_data


async def supervisor(state: DeepResearchState) -> dict[str, Any]:
    """Supervisor node with structured output and fallback."""
    llm = get_llm(ModelTier.OPUS)

    # Try structured output first (more reliable)
    try:
        action, action_data = await _get_supervisor_decision_structured(
            llm, SUPERVISOR_SYSTEM_CACHED, user_prompt, brief
        )
    except Exception as e:
        logger.warning(f"Structured output failed, falling back to text parsing: {e}")
        # Fall back to improved text parsing
        action, action_data = _parse_supervisor_response(content, brief)

    return {"action": action, "action_data": action_data}
```

### Step 5: Add Depth-Based Recursion Limits

Configured recursion limits to prevent unbounded iterations:

```python
# workflows/research/graph.py

RECURSION_LIMITS = {
    "quick": 50,
    "standard": 100,
    "comprehensive": 200,
}

def build_research_graph(depth: str = "standard") -> StateGraph:
    """Build research workflow with appropriate recursion limit."""
    graph = StateGraph(DeepResearchState)
    # ... build graph ...

    return graph.compile(
        recursion_limit=RECURSION_LIMITS.get(depth, 100)
    )
```

## Prevention

### Prompt Design Guidelines

1. **Separate operational metadata with clear boundaries**:
   ```python
   # GOOD
   """
   <Operational Metadata - DO NOT reference in outputs>
   Iteration: {iteration}/{max}
   Completeness: {score}%
   </Operational Metadata>

   Research Topic: {topic}
   Generate questions about the research topic only.
   """
   ```

2. **Explicit non-inclusion instructions**: Always tell the LLM what NOT to include

3. **Use structured output**: `with_structured_output()` is far more reliable than JSON regex extraction

4. **Defense in depth**: Validate LLM outputs even after structured extraction

5. **Fail open with logging**: When validation fails, accept with warning rather than blocking

## Files Modified

- `workflows/research/state.py`: Added Pydantic models for structured output
- `workflows/research/subgraphs/researcher.py`: Structured query generation with validation
- `workflows/research/nodes/supervisor.py`: Structured supervisor decision with fallback
- `workflows/research/prompts.py`: Hardened prompts with metadata separation
- `workflows/research/graph.py`: Added depth-based recursion limits

## Related Patterns

- [Anthropic Claude Integration](../../patterns/llm-interaction/anthropic-claude-extended-thinking.md) - Structured output patterns
- [Deep Research Workflow](../../patterns/langgraph/deep-research-workflow-architecture.md) - Workflow architecture

## References

- [LangChain Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic v2 Field Descriptions](https://docs.pydantic.dev/latest/concepts/fields/)
