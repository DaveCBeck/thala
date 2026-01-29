---
name: multi-angle-content-generation
title: "Multi-Angle Content Generation with Parallel Evaluation"
date: 2026-01-14
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/651282f6f5374b26b08ae8be2d9327a4
article_path: .context/libs/thala-dev/content/2026-01-14-multi-angle-content-generation-langgraph.md
applicability:
  - "Long-form content generation needing multiple perspectives"
  - "Essay writing with uncertain best framing approach"
  - "Output quality critical enough to justify parallel generation cost"
  - "Content transformation (e.g., academic review → engaging essay)"
components: [parallel_writers, angle_prompts, cross_evaluator, state_aggregation]
complexity: moderate
verified_in_production: false
deprecated: true
deprecated_date: 2026-01-29
deprecated_reason: "Workflow directory (workflows/output/substack_review/) was never created; pattern describes unimplemented design"
related_solutions: []
tags: [parallel-generation, content-generation, evaluation, opus, essay-writing, send-pattern]
---

> **DEPRECATED**: This documentation describes a workflow that was never implemented.
>
> **Reason:** The `workflows/output/substack_review/` directory does not exist. The pattern describes a design that was not built.
> **Date:** 2026-01-29
> **See instead:** [Content Series Generation](./content-series-generation.md) for a similar parallel generation pattern that IS implemented

# Multi-Angle Content Generation with Parallel Evaluation

## Intent

Generate multiple content variants with distinct framing angles in parallel, then use structured cross-evaluation to select the highest-quality output.

## Motivation

Single-agent content generation has blind spots—the first framing chosen may not be the best. Iterating sequentially wastes time and may not explore fundamentally different approaches.

This pattern solves both problems:
- **Parallel generation**: Three distinct angles generate simultaneously (no latency penalty)
- **Cross-evaluation**: An evaluator agent compares all variants with structured criteria
- **Traceability**: Full evaluation scores and reasoning preserved for analysis

## Applicability

Use this pattern when:
- Output quality justifies 3x generation cost (essay publishing, important reports)
- Multiple valid framing approaches exist for the source material
- Content benefits from different narrative structures (puzzle vs. evidence vs. contrarian)
- Selection criteria can be articulated as structured dimensions

Do NOT use this pattern when:
- Simple, formulaic output is acceptable
- Cost constraints preclude multiple generations
- Single "correct" framing is obvious
- Output will be heavily edited anyway

## Structure

```
Source Material (Literature Review)
              │
              ▼
       ┌──────────────────────────────┐
       │      validate_input          │
       └──────────────────────────────┘
              │
              ▼ (Send pattern - parallel fan-out)
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌────────┐┌────────┐┌────────┐
│ PUZZLE ││FINDING ││CONTRAR-│
│ Writer ││ Writer ││  IAN   │
└────┬───┘└────┬───┘└────┬───┘
     │         │         │
     └─────────┼─────────┘
              ▼ (State aggregation)
       ┌──────────────────────────────┐
       │      choose_essay            │
       │  (6-dimension evaluation)    │
       └──────────────────────────────┘
              │
              ▼
       Selected Essay + Evaluation Scores
```

## Implementation

### Step 1: Define Angle-Specific Prompts

Each angle has a distinct narrative purpose:

```python
# prompts.py

PUZZLE_SYSTEM_PROMPT = """You are transforming a literature review into an
engaging essay using the PUZZLE angle.

**Structure:**
1. Open with a specific, surprising detail from the literature
2. Unfold investigation around that puzzle
3. Use the puzzle as lens to understand broader themes
4. Surface tensions and unresolved questions

**Tone:** Curious, investigative, intellectually honest
**Best for:** Readers who want to follow an unfolding argument
"""

FINDING_SYSTEM_PROMPT = """You are transforming a literature review into an
engaging essay using the FINDING angle.

**Structure:**
1. Open with the most surprising quantitative result
2. Explain what this tells us about prior assumptions
3. Walk through the mechanism that produced the result
4. Expand to related themes and implications

**Tone:** Direct, energetic, intellectually engaged
**Best for:** Readers seeking concrete evidence
"""

CONTRARIAN_SYSTEM_PROMPT = """You are transforming a literature review into an
engaging essay using the CONTRARIAN angle.

**Structure:**
1. Articulate the comfortable assumption (steelman it)
2. Introduce the complication—what doesn't fit?
3. Work through evidence that complicates the simple story
4. Close with productive uncertainty

**Tone:** Thoughtful, fair, precise
**Best for:** Readers who value nuance
"""
```

### Step 2: Create Parallel Writing Nodes

```python
# nodes/write_essay.py

from typing import Literal, Any
from langchain_anthropic import ChatAnthropic
from workflows.shared.llm_utils import ModelTier


async def _write_essay(
    literature_review: str,
    angle: Literal["puzzle", "finding", "contrarian"],
    system_prompt: str,
    user_template: str,
) -> EssayDraft:
    """Write a single essay with the given angle."""
    llm = ChatAnthropic(
        model=ModelTier.OPUS.value,
        max_tokens=16000,  # 3000-4000 words
    )

    user_prompt = user_template.format(literature_review=literature_review)

    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)

    return EssayDraft(
        angle=angle,
        content=content,
        word_count=len(content.split()),
    )


async def write_puzzle_essay(state: dict) -> dict[str, Any]:
    """Write essay using Puzzle angle."""
    lit_review = state.get("literature_review") or state["input"]["literature_review"]

    try:
        essay = await _write_essay(
            literature_review=lit_review,
            angle="puzzle",
            system_prompt=PUZZLE_SYSTEM_PROMPT,
            user_template=PUZZLE_USER_TEMPLATE,
        )
        return {"essay_drafts": [essay]}
    except Exception as e:
        return {"errors": [{"node": "write_puzzle_essay", "error": str(e)}]}


# Similarly: write_finding_essay(), write_contrarian_essay()
```

### Step 3: Define State with Aggregation

```python
# state.py

from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict


class EssayDraft(TypedDict):
    angle: Literal["puzzle", "finding", "contrarian"]
    content: str
    word_count: int


class SubstackReviewState(TypedDict):
    # Input
    input: EssayInput

    # Writing phase - aggregates from parallel nodes
    essay_drafts: Annotated[list[EssayDraft], add]

    # Selection phase
    selected_angle: Optional[Literal["puzzle", "finding", "contrarian"]]
    selection_reasoning: Optional[str]
    essay_evaluations: Optional[dict]

    # Output
    final_essay: Optional[str]
    errors: Annotated[list[dict], add]
```

The `Annotated[list[EssayDraft], add]` reducer merges outputs from parallel nodes into a single list.

### Step 4: Build Graph with Parallel Fan-Out

```python
# graph.py

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


def route_after_validation(state: SubstackReviewState) -> list[Send] | str:
    """Route to parallel writing agents."""
    if not state.get("is_valid"):
        return END

    lit_review = state["input"]["literature_review"]

    # Fan out to 3 parallel writing agents
    return [
        Send("write_puzzle", {"literature_review": lit_review, "input": state["input"]}),
        Send("write_finding", {"literature_review": lit_review, "input": state["input"]}),
        Send("write_contrarian", {"literature_review": lit_review, "input": state["input"]}),
    ]


def create_substack_review_graph() -> StateGraph:
    builder = StateGraph(SubstackReviewState)

    builder.add_node("validate_input", validate_input_node)
    builder.add_node("write_puzzle", write_puzzle_essay)
    builder.add_node("write_finding", write_finding_essay)
    builder.add_node("write_contrarian", write_contrarian_essay)
    builder.add_node("choose_essay", choose_essay_node)

    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        ["write_puzzle", "write_finding", "write_contrarian", END],
    )

    # All writers converge to chooser
    builder.add_edge("write_puzzle", "choose_essay")
    builder.add_edge("write_finding", "choose_essay")
    builder.add_edge("write_contrarian", "choose_essay")
    builder.add_edge("choose_essay", END)

    return builder.compile()
```

### Step 5: Implement Cross-Evaluation

```python
# schemas.py

from pydantic import BaseModel, Field
from typing import Literal


class EssayEvaluation(BaseModel):
    """Evaluation of a single essay."""
    primary_strength: str
    primary_weakness: str
    hook_strength: int = Field(ge=1, le=5)
    structural_momentum: int = Field(ge=1, le=5)
    technical_payoff: int = Field(ge=1, le=5)
    tonal_calibration: int = Field(ge=1, le=5)
    honest_complexity: int = Field(ge=1, le=5)
    subject_fit: int = Field(ge=1, le=5)


class ChoosingAgentOutput(BaseModel):
    """Structured output from the choosing agent."""
    selected: Literal["puzzle", "finding", "contrarian"]
    evaluations: dict[str, EssayEvaluation]
    deciding_factors: str = Field(description="2-3 sentences on what made winner stand out")
    close_call: bool = Field(default=False)
    close_call_explanation: str = Field(default="")
```

```python
# nodes/choose_essay.py

async def choose_essay_node(state: SubstackReviewState) -> dict[str, Any]:
    """Select the best essay using structured evaluation."""
    essay_drafts = state.get("essay_drafts", [])
    essays_by_angle = {e["angle"]: e["content"] for e in essay_drafts}

    # Verify all three present
    expected = {"puzzle", "finding", "contrarian"}
    missing = expected - set(essays_by_angle.keys())
    if missing:
        return {"status": "failed", "errors": [{"error": f"Missing: {missing}"}]}

    user_prompt = CHOOSING_USER_TEMPLATE.format(
        essay_puzzle=essays_by_angle["puzzle"],
        essay_finding=essays_by_angle["finding"],
        essay_contrarian=essays_by_angle["contrarian"],
    )

    result: ChoosingAgentOutput = await get_structured_output(
        output_schema=ChoosingAgentOutput,
        user_prompt=user_prompt,
        system_prompt=CHOOSING_SYSTEM_PROMPT,
        tier=ModelTier.OPUS,
    )

    return {
        "selected_angle": result.selected,
        "selection_reasoning": result.deciding_factors,
        "essay_evaluations": {
            angle: eval.model_dump()
            for angle, eval in result.evaluations.items()
        },
    }
```

## Complete Example

```python
# Running the workflow

from workflows.output.substack_review import substack_review

result = await substack_review(
    literature_review=lit_review_text,
    title="AI Safety Research",
    target_word_count=3500,
)

print(f"Selected angle: {result['selected_angle']}")
print(f"Reasoning: {result['selection_reasoning']}")
print(f"Word count: {len(result['final_essay'].split())}")

# Access detailed evaluations
for angle, scores in result["essay_evaluations"].items():
    print(f"\n{angle.upper()}:")
    print(f"  Hook: {scores['hook_strength']}/5")
    print(f"  Momentum: {scores['structural_momentum']}/5")
    print(f"  Payoff: {scores['technical_payoff']}/5")
```

## Consequences

### Benefits

- **Quality improvement**: Selection from 3 variants beats single-shot generation
- **No latency penalty**: Parallel execution means 1x time cost (not 3x)
- **Explainable selection**: Structured scores justify the choice
- **Angle diversity**: Each framing serves different reader preferences
- **Close-call detection**: Flag indicates when alternatives were competitive

### Trade-offs

- **3x generation cost**: Three OPUS calls for writing (mitigated by batch API)
- **Complexity**: More nodes and state management than single-agent
- **Prompt engineering**: Each angle needs careful prompt design
- **Evaluation calibration**: Scoring dimensions may need tuning

### Alternatives

- **Sequential iteration**: Generate one, evaluate, refine (slower)
- **A/B with human selection**: Generate two, let user choose (simpler)
- **Single-shot with angle parameter**: User pre-selects angle (cheapest)

## Related Patterns

- [Parallel AI Search Integration](../data-pipeline/parallel-ai-search-integration.md) - Parallel execution with fault tolerance
- [Specialized Researcher Pattern](./specialized-researcher-pattern.md) - Source-specific parallel agents
- [Deep Research Workflow Architecture](./deep-research-workflow-architecture.md) - Send() pattern for fan-out

## Known Uses in Thala

- `workflows/output/substack_review/graph.py` - Substack essay generation
- `workflows/output/substack_review/nodes/write_essay.py` - Parallel angle writers
- `workflows/output/substack_review/nodes/choose_essay.py` - Cross-evaluation selection
- `workflows/output/substack_review/schemas.py` - EssayEvaluation Pydantic model

## References

- [LangGraph Send Pattern](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [Structured Output with Pydantic](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
