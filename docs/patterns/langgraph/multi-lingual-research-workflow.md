---
name: multi-lingual-research-workflow
title: "Multi-Lingual Research Workflow"
date: 2025-12-22
category: langgraph
shared: true
gist_url: "https://gist.github.com/DaveCBeck/d97b27ed37ec6166d2a6b4d4bba8d06b"
article_path: ".context/libs/thala-dev/content/2025-12-22-multi-lingual-research-workflow-langgraph.md"
applicability:
  - "When research topics have language-specific sources offering unique insights"
  - "When cross-cultural analysis requires perspectives from different language communities"
  - "When translating final output while preserving academic citations"
components: [langgraph_node, langgraph_state, llm_call, workflow_graph]
complexity: complex
verified_in_production: false
deprecated: true
deprecated_date: 2026-01-29
deprecated_reason: "Features described (analyze_languages, synthesize_languages, translate_report nodes) were never implemented"
superseded_by: "./multi-language-workflow-orchestration.md"
related_solutions: []
tags: [langgraph, multi-lingual, research, translation, cross-cultural, conditional-routing, parallel-agents, round-robin]
---

> **DEPRECATED**: This documentation describes features that were never implemented.
>
> **Reason:** The nodes described (analyze_languages, synthesize_languages, translate_report) do not exist in the codebase. This was a design document, not implementation documentation.
> **Date:** 2026-01-29
> **See instead:** [Multi-Language Workflow Orchestration](./multi-language-workflow-orchestration.md) for the actual implementation

# Multi-Lingual Research Workflow

## Intent

Enable research workflows to operate across multiple languages, with automatic language selection, language-specific researcher agents, cross-language synthesis of findings, and optional translation of final output.

## Motivation

Many research topics benefit from sources in multiple languages:

1. **Regional expertise**: Medical research in German journals, fashion in French publications
2. **Cultural perspectives**: Same topic covered differently across cultures
3. **Academic specialization**: Fields where non-English literature is dominant
4. **Market research**: Understanding local markets requires local-language sources

A single-language workflow misses these insights. This pattern adds:
- Automatic language selection via LLM analysis
- Language-aware researcher agents distributed across languages
- Cross-language synthesis identifying unique cultural insights
- Optional translation preserving citations and quotes

## Applicability

Use this pattern when:
- Research topics span multiple cultural or regional contexts
- Non-English sources provide unique expertise (academic journals, local news)
- Cross-cultural comparison is valuable
- Final output needs translation while preserving academic formatting

Do NOT use this pattern when:
- Topic is well-covered in English (no unique insights elsewhere)
- Speed is critical (adds latency for analysis/synthesis/translation)
- Single target audience with known language preference

## Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Lingual Research Workflow                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              START
                                │
                                ▼
                        ┌───────────────┐
                        │ clarify_intent │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ create_brief  │
                        └───────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │  multi_lingual=True?  │
                    └───────────┬───────────┘
                        Yes │       │ No
                            ▼       │
                ┌─────────────────┐ │
                │analyze_languages│ │
                │  (LLM-based)    │ │
                └────────┬────────┘ │
                         │          │
                         ▼          │
              ┌──────────────────┐  │
              │ active_languages │  │
              │ language_configs │  │
              └────────┬─────────┘  │
                       │            │
                       └────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ search_memory │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  supervisor   │◄───────────────────┐
                    └───────┬───────┘                    │
                            │                            │
            ┌───────────────┼───────────────┐            │
            ▼               ▼               ▼            │
    ┌──────────────┐┌──────────────┐┌──────────────┐     │
    │ Researcher   ││ Researcher   ││ Researcher   │     │
    │ (Spanish)    ││ (German)     ││ (Chinese)    │     │
    │ lang_config  ││ lang_config  ││ lang_config  │     │
    └──────┬───────┘└──────┬───────┘└──────┬───────┘     │
           │               │               │             │
           └───────────────┼───────────────┘             │
                           │                             │
                           ▼                             │
                ┌────────────────────┐                   │
                │aggregate_findings  │                   │
                │ (groups by lang)   │───────────────────┘
                └────────┬───────────┘
                         │
            ┌────────────┴────────────┐
            │  research_complete?     │
            └────────────┬────────────┘
                 Yes │       │ No (loop)
                     ▼
           ┌──────────────────┐
           │synthesize_langs  │  (if multi_lingual)
           │(cross-cultural)  │
           └────────┬─────────┘
                    │
                    ▼
            ┌───────────────┐
            │ final_report  │
            └───────┬───────┘
                    │
            ┌───────┴───────┐
            │translate_to?  │
            └───────┬───────┘
                Yes │
                    ▼
           ┌───────────────┐
           │translate_report│
           │(preserve cites)│
           └───────┬───────┘
                   │
                   ▼
                  END
```

## Implementation

### Step 1: Define Language Configuration Types

Add TypedDict types for language configuration in state:

```python
# workflows/research/state.py

class LanguageConfig(TypedDict):
    """Configuration for a specific language in the research workflow."""

    code: str  # ISO 639-1 code (e.g., "es", "zh", "ja")
    name: str  # Full language name (e.g., "Spanish", "Mandarin Chinese")
    search_domains: list[str]  # Preferred domain TLDs (e.g., [".es", ".mx"])
    search_engine_locale: str  # Locale code for search APIs (e.g., "es-ES")


class TranslationConfig(TypedDict):
    """Configuration for translating the final research output."""

    enabled: bool  # Whether to translate the final report
    target_language: str  # Target language code (e.g., "en")
    preserve_quotes: bool  # Keep direct quotes in original language
    preserve_citations: bool  # Keep citation format unchanged
```

### Step 2: Extend ResearcherState with Language Config

```python
# workflows/research/state.py

class ResearcherState(TypedDict):
    question: ResearchQuestion
    search_queries: list[str]
    search_results: list[WebSearchResult]
    scraped_content: list[str]
    thinking: Optional[str]
    finding: Optional[ResearchFinding]
    research_findings: Annotated[list[ResearchFinding], add]

    # Language configuration for multi-lingual support
    language_config: Optional[LanguageConfig]


class ResearchFinding(TypedDict):
    # ... existing fields ...
    language_code: Optional[str]  # ISO 639-1 code for findings grouping
```

### Step 3: Extend Main Workflow State

```python
# workflows/research/state.py

class DeepResearchState(TypedDict):
    # ... existing fields ...

    # Multi-lingual support
    primary_language: Optional[str]  # Single-language mode
    primary_language_config: Optional[LanguageConfig]
    active_languages: Optional[list[str]]  # Languages being researched
    language_configs: Optional[dict[str, LanguageConfig]]  # Config per language
    language_findings: Optional[dict[str, list[ResearchFinding]]]  # Findings by language
    language_synthesis: Optional[str]  # Cross-language synthesis

    # Translation
    translation_config: Optional[TranslationConfig]
    translated_report: Optional[str]
```

### Step 4: Create Language Analysis Node

Use structured output to recommend languages:

```python
# workflows/research/nodes/analyze_languages.py

class LanguageRecommendation(BaseModel):
    """Recommendation for a specific language to research."""
    language_code: str = Field(description="ISO 639-1 language code")
    rationale: str = Field(description="Why this language adds unique value")
    expected_unique_insights: list[str]
    priority: int = Field(ge=1)


class LanguageAnalysisResult(BaseModel):
    """Result of language analysis for multi-lingual research."""
    recommendations: list[LanguageRecommendation] = Field(max_length=5)
    reasoning: str


ANALYZE_LANGUAGES_SYSTEM = """You are an expert in identifying language-specific research opportunities.

Analyze this research topic and recommend which languages would provide UNIQUE, valuable insights that aren't readily available in English sources.

CRITICAL GUIDELINES:
- Only recommend languages offering genuinely unique perspectives
- Do NOT recommend a language just because it's widely spoken
- Consider: academic conferences, specialized journals, regional expertise, cultural perspectives
- Maximum 3-4 languages unless exceptionally global topic
- Priority 1 = most valuable"""


async def analyze_languages(state: DeepResearchState) -> dict[str, Any]:
    """Analyze which languages provide unique insights for this topic."""
    input_data = state["input"]

    # Skip if not multi-lingual mode
    if not input_data.get("multi_lingual"):
        return {"current_status": state.get("current_status")}

    # If target_languages pre-specified, use those
    if target_languages := input_data.get("target_languages"):
        language_configs = {
            code: get_language_config(code)
            for code in target_languages
            if get_language_config(code)
        }
        return {
            "active_languages": list(language_configs.keys()),
            "language_configs": language_configs,
            "current_status": "languages_analyzed",
        }

    # Otherwise use LLM to recommend
    llm = get_llm(ModelTier.OPUS)
    structured_llm = llm.with_structured_output(LanguageAnalysisResult)

    result = await structured_llm.ainvoke([
        {"role": "system", "content": ANALYZE_LANGUAGES_SYSTEM},
        {"role": "user", "content": format_analysis_prompt(state)},
    ])

    # Build configs for recommended languages
    language_configs = {}
    for rec in result.recommendations:
        if config := get_language_config(rec.language_code):
            language_configs[rec.language_code] = config

    return {
        "active_languages": list(language_configs.keys()),
        "language_configs": language_configs,
        "current_status": "languages_analyzed",
    }
```

### Step 5: Implement Round-Robin Question Distribution

Distribute questions across languages in supervisor routing:

```python
# workflows/research/graph.py

def route_supervisor_action(state: DeepResearchState) -> str | list[Send]:
    """Route based on supervisor's chosen action."""
    current_status = state.get("current_status", "")

    if current_status == "conduct_research":
        pending = state.get("pending_questions", [])
        if not pending:
            return "synthesize_languages" if state["input"].get("multi_lingual") else "final_report"

        # Multi-lingual mode: round-robin distribution
        if state["input"].get("multi_lingual") and state.get("language_configs"):
            language_configs = state.get("language_configs", {})
            active_languages = list(language_configs.keys())

            researchers = []
            for i, q in enumerate(pending):
                # Round-robin across languages
                target_lang = active_languages[i % len(active_languages)]
                lang_config = language_configs.get(target_lang)

                researchers.append(
                    Send("researcher", ResearcherState(
                        question=q,
                        language_config=lang_config,
                        # ... other fields
                    ))
                )
            return researchers
        else:
            # Single-language mode
            language_config = state.get("primary_language_config")
            return [
                Send("researcher", ResearcherState(
                    question=q,
                    language_config=language_config,
                ))
                for q in pending
            ]

    elif current_status == "research_complete":
        if state["input"].get("multi_lingual") and state.get("active_languages"):
            return "synthesize_languages"
        return "final_report"
```

### Step 6: Aggregate Findings by Language

Group findings by language code for synthesis:

```python
# workflows/research/graph.py

def aggregate_researcher_findings(state: DeepResearchState) -> dict[str, Any]:
    """Aggregate findings, grouping by language for multi-lingual mode."""
    findings = state.get("findings", [])
    input_data = state.get("input", {})

    # Group findings by language
    language_findings = None
    if input_data.get("multi_lingual") and findings:
        language_findings = {}
        for f in findings:
            lang_code = f.get("language_code") or "en"
            if lang_code not in language_findings:
                language_findings[lang_code] = []
            language_findings[lang_code].append(f)

    result = {"pending_questions": [], "current_status": "supervising"}

    if language_findings is not None:
        result["language_findings"] = language_findings

    return result
```

### Step 7: Create Cross-Language Synthesis Node

Synthesize unique insights across languages:

```python
# workflows/research/nodes/synthesize_languages.py

SYNTHESIS_SYSTEM = """You are synthesizing research findings from multiple languages.

For each language stream, identify:
1. UNIQUE insights not found in other languages
2. Cultural or regional perspectives
3. Consensus across languages (confirms findings)
4. Contradictions requiring resolution

Be specific about which insights come from which language sources."""


async def synthesize_languages(state: DeepResearchState) -> dict[str, Any]:
    """Synthesize findings across language streams."""
    language_findings = state.get("language_findings", {})

    if not language_findings or len(language_findings) < 2:
        return {"current_status": "synthesizing_complete"}

    llm = get_llm(ModelTier.OPUS)

    # Format findings by language
    formatted = []
    for lang_code, findings in language_findings.items():
        formatted.append(f"## {LANGUAGE_NAMES.get(lang_code, lang_code)} Sources\n")
        for f in findings:
            formatted.append(f"- {f['summary']}\n")

    response = await llm.ainvoke([
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {"role": "user", "content": "\n".join(formatted)},
    ])

    return {
        "language_synthesis": response.content,
        "current_status": "synthesizing_complete",
    }
```

### Step 8: Create Translation Node

Translate final report with preservation options:

```python
# workflows/research/nodes/translate_report.py

TRANSLATION_SYSTEM = """You are an expert academic translator.

Translate the following research report to {target_language}.

CRITICAL REQUIREMENTS:
- Maintain academic tone and precision
- Preserve all citation references exactly as written (e.g., [1], [2])
{quote_instruction}
- Keep proper nouns, technical terms, and acronyms as appropriate
- Maintain paragraph structure and heading hierarchy"""


async def translate_report(state: DeepResearchState) -> dict[str, Any]:
    """Translate final report to target language."""
    translation_config = state.get("translation_config")
    final_report = state.get("final_report")

    if not translation_config or not translation_config.get("enabled"):
        return {}

    if not final_report:
        return {"errors": [{"node": "translate_report", "error": "No report to translate"}]}

    target_lang = translation_config.get("target_language", "en")
    preserve_quotes = translation_config.get("preserve_quotes", True)

    quote_instruction = (
        "- Keep direct quotes in their original language, with translation in parentheses"
        if preserve_quotes
        else "- Translate all quotes to the target language"
    )

    llm = get_llm(ModelTier.OPUS)

    response = await llm.ainvoke([
        {"role": "system", "content": TRANSLATION_SYSTEM.format(
            target_language=LANGUAGE_NAMES.get(target_lang, target_lang),
            quote_instruction=quote_instruction,
        )},
        {"role": "user", "content": final_report},
    ])

    return {"translated_report": response.content}
```

### Step 9: Wire Up Graph with Conditional Routing

```python
# workflows/research/graph.py

def route_after_create_brief(state: DeepResearchState) -> str:
    """Route to language analysis if multi-lingual, otherwise to memory search."""
    if state["input"].get("multi_lingual"):
        return "analyze_languages"
    return "search_memory"


def route_after_final_report(state: DeepResearchState) -> str:
    """Route to translation if configured."""
    if state.get("translation_config", {}).get("enabled"):
        return "translate_report"
    return "process_citations"


def create_deep_research_graph():
    builder = StateGraph(DeepResearchState)

    # Add nodes
    builder.add_node("clarify_intent", clarify_intent)
    builder.add_node("create_brief", create_brief)
    builder.add_node("analyze_languages", analyze_languages)  # New
    builder.add_node("search_memory", search_memory_node)
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher_subgraph)
    builder.add_node("aggregate_findings", aggregate_researcher_findings)
    builder.add_node("synthesize_languages", synthesize_languages)  # New
    builder.add_node("final_report", final_report)
    builder.add_node("translate_report", translate_report)  # New
    builder.add_node("process_citations", process_citations)

    # Edges with conditional routing
    builder.add_edge(START, "clarify_intent")
    builder.add_conditional_edges("create_brief", route_after_create_brief)
    builder.add_edge("analyze_languages", "search_memory")
    # ... other edges
    builder.add_conditional_edges("final_report", route_after_final_report)
    builder.add_edge("translate_report", "process_citations")

    return builder.compile()
```

## Usage Examples

```python
from workflows.research import deep_research

# Single language mode
result = await deep_research(
    topic="Impact of AI on healthcare in Japan",
    language="ja",  # Research in Japanese
    translate_to="en",  # Translate final report to English
    preserve_quotes=True,  # Keep Japanese quotes with translations
)

# Multi-lingual mode (automatic language selection)
result = await deep_research(
    topic="Climate change policies",
    multi_lingual=True,  # Auto-select relevant languages
    translate_to="en",
)

# Multi-lingual mode (explicit languages)
result = await deep_research(
    topic="Supply chain disruptions",
    multi_lingual=True,
    target_languages=["zh", "de", "ja"],  # Specific languages
)
```

## Consequences

### Benefits

- **Unique insights**: Access language-specific sources unavailable in English
- **Cross-cultural synthesis**: Identify consensus and cultural differences
- **Flexible modes**: Single-language, auto-select, or explicit language lists
- **Citation preservation**: Academic formatting maintained through translation

### Trade-offs

- **Increased latency**: Language analysis, synthesis, and translation add time
- **Higher cost**: Multiple LLM calls for analysis/synthesis/translation
- **Complexity**: Conditional routing and state aggregation add complexity
- **Translation quality**: Even Opus can miss nuance in specialized domains

## Related Patterns

- [Deep Research Workflow Architecture](./deep-research-workflow-architecture.md) - Base workflow this extends
- [Parallel AI Search Integration](../data-pipeline/parallel-ai-search-integration.md) - Parallel execution pattern

## Known Uses in Thala

- `workflows/research/graph.py`: Graph construction with language routing
- `workflows/research/nodes/analyze_languages.py`: LLM-based language recommendation
- `workflows/research/nodes/synthesize_languages.py`: Cross-language synthesis
- `workflows/research/nodes/translate_report.py`: Report translation
- `workflows/research/config/languages.py`: Language configuration registry

## References

- [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
- [LangGraph Send API](https://python.langchain.com/docs/langgraph/concepts/low_level/#send)
