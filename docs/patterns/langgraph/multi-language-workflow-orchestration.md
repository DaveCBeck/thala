---
name: multi-language-workflow-orchestration
title: Multi-Language Workflow Orchestration Pattern
date: 2026-01-02
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/1a87233486fe5d392d9e5732be23bf3d
article_path: .context/libs/thala-dev/content/2026-01-02-multi-language-workflow-orchestration-langgraph.md
applicability:
  - "Research requiring perspectives from multiple language communities"
  - "Cross-cultural analysis and synthesis"
  - "Auto-detecting relevant languages for a topic"
  - "Producing both integrated and comparative documents"
components: [multi_lang, language_selector, relevance_checker, opus_integrator, sonnet_analyzer]
complexity: high
verified_in_production: true
tags: [multi-language, orchestration, synthesis, opus, sonnet, haiku, cross-cultural, research]
---

# Multi-Language Workflow Orchestration Pattern

## Intent

Run research workflows across multiple languages, filter by relevance, and produce two outputs: a **synthesized document** (Opus integrates findings one-by-one) and a **comparative document** (Sonnet 1M analyzes cross-language patterns).

## Problem

Research in a single language misses:
- Regional perspectives and local expertise
- Non-English academic literature
- Cultural variations in approach
- Coverage gaps in English-language sources

Running research in multiple languages creates new challenges:
- Determining which languages are relevant
- Managing sequential execution with checkpointing
- Synthesizing findings into coherent output
- Identifying cross-language patterns

## Solution

Create a workflow that:
1. **Selects languages**: Either explicit list or auto-detect from major 10
2. **Filters by relevance**: Haiku-powered relevance check per language
3. **Executes per-language**: Sequential with checkpointing
4. **Analyzes patterns**: Sonnet 1M produces comparative document
5. **Integrates findings**: Opus integrates one-by-one into synthesis
6. **Saves with lineage**: Store records track source languages

## Structure

```
workflows/multi_lang/
├── __init__.py           # Public API exports
├── state.py              # TypedDict states with relevance, results, synthesis
├── checkpointing.py      # Per-language checkpoint utilities
├── prompts/
│   ├── __init__.py
│   ├── relevance.py      # Haiku relevance check prompts
│   ├── analysis.py       # Sonnet comparative analysis prompts
│   └── integration.py    # Opus synthesis prompts
├── graph/
│   ├── __init__.py
│   ├── construction.py   # LangGraph builder
│   ├── routing.py        # Language iteration routing
│   └── api.py            # multi_lang_research() entry point
└── nodes/
    ├── __init__.py
    ├── language_selector.py    # Set/auto-detect languages
    ├── relevance_checker.py    # Haiku batch relevance filter
    ├── language_executor.py    # Run research per language
    ├── sonnet_analyzer.py      # Cross-language comparison
    ├── opus_integrator.py      # One-by-one synthesis
    └── save_results.py         # Store with lineage
```

## Implementation

### State Definition

```python
# workflows/multi_lang/state.py

from typing import Literal, Optional, Annotated
from typing_extensions import TypedDict
from operator import add


class LanguageRelevanceCheck(TypedDict):
    """Haiku-powered relevance decision for a language."""
    language_code: str
    has_meaningful_discussion: bool
    confidence: float  # 0-1
    reasoning: str
    suggested_depth: Literal["skip", "quick", "standard", "comprehensive"]


class LanguageResult(TypedDict):
    """Results from research in one language."""
    language_code: str
    language_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    workflows_run: list[str]  # ["web", "academic", "books"]
    quality_used: str
    findings_summary: str
    full_report: Optional[str]
    source_count: int
    key_insights: list[str]
    unique_perspectives: list[str]
    store_record_id: Optional[str]
    errors: list[dict]


class SonnetCrossAnalysis(TypedDict):
    """Sonnet-powered comparative analysis across languages."""
    # Commonalities
    universal_themes: list[str]
    consensus_findings: list[str]
    # Differences
    regional_variations: list[dict]  # {theme, variations: [{language, perspective}]}
    conflicting_findings: list[dict]
    unique_contributions: dict[str, list[str]]  # language_code -> insights
    # Coverage
    coverage_gaps_in_english: list[str]
    enhanced_areas: list[str]
    # For Opus integration
    integration_priority: list[str]  # ordered language codes
    synthesis_strategy: str
    # Output
    comparative_document: str


class MultiLangInput(TypedDict):
    """Input parameters for multi-language research."""
    topic: str
    research_questions: Optional[list[str]]
    brief: Optional[str]
    mode: Literal["set_languages", "all_languages"]
    languages: Optional[list[str]]  # ISO 639-1 codes
    workflows: WorkflowSelection
    quality_settings: MultiLangQualitySettings


class CheckpointPhase(TypedDict):
    """Tracks completion for resumption."""
    language_selection: bool
    relevance_checks: bool
    languages_executed: dict[str, bool]  # code -> completed
    sonnet_analysis: bool
    opus_integration: bool
    saved_to_store: bool


class MultiLangState(TypedDict):
    """Complete state for multi-language orchestration."""
    input: MultiLangInput

    # Language selection
    target_languages: list[str]  # All languages to consider
    relevance_checks: list[LanguageRelevanceCheck]
    languages_with_content: list[str]  # Filtered relevant languages

    # Per-language results
    language_results: Annotated[list[LanguageResult], add]
    current_language_index: int

    # Synthesis outputs
    sonnet_analysis: Optional[SonnetCrossAnalysis]
    opus_integration_steps: list[OpusIntegrationStep]
    final_synthesis: Optional[str]

    # Store integration
    store_record_ids: dict[str, str]  # type -> UUID

    # Checkpointing
    checkpoint_phase: CheckpointPhase

    # Metadata
    started_at: datetime
    completed_at: Optional[datetime]
    errors: Annotated[list[dict], add]
```

### Major 10 Languages for Auto-Detection

```python
# workflows/multi_lang/nodes/language_selector.py

MAJOR_LANGUAGES = [
    "en",  # English
    "zh",  # Chinese
    "es",  # Spanish
    "ar",  # Arabic
    "hi",  # Hindi
    "pt",  # Portuguese
    "ja",  # Japanese
    "de",  # German
    "fr",  # French
    "ru",  # Russian
]


async def select_languages(state: MultiLangState) -> dict[str, Any]:
    """Select target languages based on mode."""
    input_data = state["input"]
    mode = input_data["mode"]

    if mode == "set_languages":
        # Explicit list provided
        target_languages = input_data.get("languages", ["en"])
    else:
        # all_languages mode - use major 10
        target_languages = MAJOR_LANGUAGES.copy()

    return {
        "target_languages": target_languages,
        "current_phase": "language_selection_complete",
    }
```

### Haiku Relevance Filter

```python
# workflows/multi_lang/nodes/relevance_checker.py

from workflows.shared.llm_utils import get_llm, ModelTier

RELEVANCE_CHECK_PROMPT = """Determine if the language "{language_name}" has meaningful unique content for the topic: "{topic}"

Consider:
1. Is this language spoken in regions with expertise on this topic?
2. Are there likely academic/professional discussions in this language?
3. Would this language add unique perspectives not covered in English?

Return JSON:
{{"has_meaningful_discussion": true/false, "confidence": 0.0-1.0, "reasoning": "...", "suggested_depth": "skip/quick/standard/comprehensive"}}"""


async def check_relevance_batch(state: MultiLangState) -> dict[str, Any]:
    """Check relevance for all target languages using Haiku (cheap, fast)."""
    topic = state["input"]["topic"]
    target_languages = state["target_languages"]

    llm = get_llm(ModelTier.HAIKU, max_tokens=500)
    relevance_checks = []

    for lang_code in target_languages:
        lang_name = SUPPORTED_LANGUAGES[lang_code]["name"]

        prompt = RELEVANCE_CHECK_PROMPT.format(
            language_name=lang_name,
            topic=topic,
        )

        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        result = json.loads(response.content)

        relevance_checks.append(LanguageRelevanceCheck(
            language_code=lang_code,
            has_meaningful_discussion=result["has_meaningful_discussion"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            suggested_depth=result["suggested_depth"],
        ))

    return {"relevance_checks": relevance_checks}


async def filter_relevant_languages(state: MultiLangState) -> dict[str, Any]:
    """Filter to languages with meaningful content."""
    relevance_checks = state["relevance_checks"]

    languages_with_content = [
        check["language_code"]
        for check in relevance_checks
        if check["has_meaningful_discussion"] and check["suggested_depth"] != "skip"
    ]

    logger.info(f"Filtered to {len(languages_with_content)}/{len(relevance_checks)} languages")

    return {
        "languages_with_content": languages_with_content,
        "current_language_index": 0,
    }
```

### Sequential Language Execution with Checkpointing

```python
# workflows/multi_lang/nodes/language_executor.py

async def execute_next_language(state: MultiLangState) -> dict[str, Any]:
    """Execute research for the next language in queue."""
    languages = state["languages_with_content"]
    index = state["current_language_index"]

    if index >= len(languages):
        return {"current_phase": "all_languages_complete"}

    lang_code = languages[index]
    lang_name = SUPPORTED_LANGUAGES[lang_code]["name"]

    logger.info(f"Executing language {index + 1}/{len(languages)}: {lang_name}")

    # Determine quality for this language
    quality = _get_quality_for_language(state, lang_code)

    # Run research workflow for this language
    try:
        result = await _run_language_research(
            topic=state["input"]["topic"],
            research_questions=state["input"].get("research_questions"),
            language_code=lang_code,
            quality=quality,
            workflows=state["input"]["workflows"],
        )

        language_result = LanguageResult(
            language_code=lang_code,
            language_name=lang_name,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            workflows_run=_get_workflows_run(state["input"]["workflows"]),
            quality_used=quality,
            findings_summary=result.get("summary", ""),
            full_report=result.get("full_report"),
            source_count=result.get("source_count", 0),
            key_insights=result.get("key_insights", []),
            unique_perspectives=result.get("unique_perspectives", []),
            store_record_id=None,
            errors=[],
        )

    except Exception as e:
        logger.error(f"Language {lang_code} failed: {e}")
        language_result = LanguageResult(
            language_code=lang_code,
            language_name=lang_name,
            # ... error fields
            errors=[{"error": str(e)}],
        )

    # Save checkpoint after each language
    save_checkpoint(state, f"language_{lang_code}")

    return {
        "language_results": [language_result],
        "current_language_index": index + 1,
    }
```

### Sonnet Comparative Analysis

```python
# workflows/multi_lang/nodes/sonnet_analyzer.py

COMPARATIVE_ANALYSIS_PROMPT = """Analyze these research findings from {num_languages} languages on the topic: "{topic}"

For each language, I'll provide a summary of findings. Your task is to:

1. COMMONALITIES: Identify universal themes and consensus findings
2. DIFFERENCES: Note regional variations and conflicting perspectives
3. UNIQUE CONTRIBUTIONS: What does each language add that others miss?
4. COVERAGE GAPS: What is poorly covered in English-only research?
5. INTEGRATION STRATEGY: How should an integrator combine these findings?

{language_findings}

Produce a detailed comparative document with all sections above."""


async def run_sonnet_analysis(state: MultiLangState) -> dict[str, Any]:
    """Produce comparative analysis using Sonnet 1M context."""
    topic = state["input"]["topic"]
    results = state["language_results"]

    # Format findings for analysis
    language_findings = _format_findings_for_analysis(results)

    llm = get_llm(ModelTier.SONNET_1M, max_tokens=8000)

    prompt = COMPARATIVE_ANALYSIS_PROMPT.format(
        num_languages=len(results),
        topic=topic,
        language_findings=language_findings,
    )

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    analysis = SonnetCrossAnalysis(
        universal_themes=_extract_themes(response.content),
        consensus_findings=_extract_consensus(response.content),
        regional_variations=_extract_variations(response.content),
        # ... other fields
        comparative_document=response.content,
    )

    save_checkpoint(state, "sonnet_analysis")

    return {"sonnet_analysis": analysis}
```

### Opus Integration

```python
# workflows/multi_lang/nodes/opus_integrator.py

INTEGRATION_PROMPT = """You are integrating research findings from {language_name} into a synthesized document.

Current synthesis so far:
{current_synthesis}

New findings from {language_name}:
{new_findings}

Sonnet's integration guidance:
{integration_strategy}

Update the synthesis by:
1. Adding new sections where {language_name} provides unique coverage
2. Enhancing existing sections with {language_name}'s perspectives
3. Noting where {language_name} confirms or contradicts existing content

Output the COMPLETE updated synthesis document."""


async def run_opus_integration(state: MultiLangState) -> dict[str, Any]:
    """Integrate findings one language at a time using Opus."""
    results = state["language_results"]
    analysis = state["sonnet_analysis"]
    priority_order = analysis["integration_priority"]

    current_synthesis = _get_initial_synthesis(state)
    integration_steps = []

    for lang_code in priority_order:
        lang_result = _get_result_for_language(results, lang_code)
        if not lang_result:
            continue

        llm = get_llm(ModelTier.OPUS, max_tokens=16000, thinking_budget=8000)

        prompt = INTEGRATION_PROMPT.format(
            language_name=lang_result["language_name"],
            current_synthesis=current_synthesis,
            new_findings=lang_result["full_report"],
            integration_strategy=analysis["synthesis_strategy"],
        )

        response = await llm.ainvoke([{"role": "user", "content": prompt}])

        current_synthesis = response.content
        integration_steps.append(OpusIntegrationStep(
            language_code=lang_code,
            language_name=lang_result["language_name"],
            integrated_content=response.content,
            # ... tracking fields
        ))

        logger.info(f"Integrated {lang_result['language_name']} into synthesis")

    save_checkpoint(state, "opus_integration")

    return {
        "opus_integration_steps": integration_steps,
        "final_synthesis": current_synthesis,
    }
```

## Usage

```python
from workflows.multi_lang import multi_lang_research

# Mode 1: Specific languages
result = await multi_lang_research(
    topic="sustainable urban planning",
    mode="set_languages",
    languages=["en", "es", "de", "ja"],
    workflows={"web": True, "academic": True, "books": False},
)

# Mode 2: Auto-detect from major 10 with Haiku filtering
result = await multi_lang_research(
    topic="traditional medicine practices",
    mode="all_languages",
    quality_settings={"default_quality": "standard"},
)

# Access outputs
print(result.synthesis)     # Opus-integrated unified report
print(result.comparative)   # Sonnet cross-language analysis
```

## Guidelines

### Mode Selection

| Mode | When to Use |
|------|-------------|
| `set_languages` | You know which languages are relevant |
| `all_languages` | Exploratory, let Haiku decide relevance |

### Quality Tiers by Language

Use per-language quality overrides:
```python
quality_settings={
    "default_quality": "standard",
    "per_language_overrides": {
        "en": {"quality_tier": "comprehensive"},  # English: most depth
        "zh": {"quality_tier": "standard"},       # Chinese: standard
        "ar": {"quality_tier": "quick"},          # Arabic: quick survey
    }
}
```

### Two-Document Output

| Document | Producer | Purpose |
|----------|----------|---------|
| Synthesized | Opus | Unified report integrating all findings |
| Comparative | Sonnet 1M | Cross-language patterns and differences |

## Known Uses

- `workflows/multi_lang/` - Full implementation
- Supports 10 major languages with Haiku relevance filtering

## Consequences

### Benefits
- **Cross-cultural perspectives**: Access non-English expertise
- **Coverage gap identification**: See what English misses
- **Structured synthesis**: Opus integrates systematically
- **Pattern discovery**: Sonnet identifies cross-language themes

### Trade-offs
- **Duration**: Sequential language execution (hours for 10 languages)
- **Cost**: Multiple LLM calls (Haiku, per-language workflows, Sonnet, Opus)
- **Complexity**: Checkpointing and state management

## Related Patterns

- [Multilingual Workflow Pattern](../llm-interaction/multilingual-workflow-pattern.md) - Single workflow with translated prompts
- [Multi-Source Research Orchestration](./multi-source-research-orchestration.md) - Parallel workflow orchestration

## References

- [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
