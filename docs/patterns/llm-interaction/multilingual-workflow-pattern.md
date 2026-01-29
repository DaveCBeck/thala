---
name: multilingual-workflow-pattern
title: Multilingual Workflow Pattern
date: 2026-01-02
category: llm-interaction
applicability:
  - "Research workflows operating in multiple languages"
  - "LLM prompts that need translation while preserving structure"
  - "Search queries for language-specific content"
  - "Maintaining English technical terms in translated prompts"
components: [language, translator, query_translator]
complexity: medium
verified_in_production: true
tags: [multilingual, translation, prompts, caching, language-config, opus]
---

# Multilingual Workflow Pattern

## Intent

Enable research workflows to operate in multiple languages by translating LLM prompts, search queries, and outputs while preserving technical terms, format placeholders, and JSON schemas.

## Problem

Research workflows hardcoded in English cannot:
- Search in other languages (missing local sources)
- Generate prompts in user's native language
- Produce final output in the desired language

Naive translation approaches break LLM prompts by:
- Translating format placeholders (`{query}` → `{consulta}`)
- Translating JSON schema keys
- Losing the instructional tone

## Solution

Create a shared language module that:
1. Defines `LanguageConfig` with search domains and locales
2. Provides **Opus-powered prompt translation** (24h cache) preserving structure
3. Provides **Haiku-powered query translation** (1h cache) for search
4. Exposes `get_language_config()` for workflow initialization
5. Adds `language` parameter to workflow APIs

## Structure

```
workflows/shared/language/
├── __init__.py           # Public API: get_language_config, translate_prompt, etc.
├── types.py              # LanguageConfig, TranslationConfig TypedDicts
├── languages.py          # SUPPORTED_LANGUAGES registry (29 languages)
├── translator.py         # Opus prompt translation with caching
└── query_translator.py   # Haiku query translation with caching
```

## Implementation

### Language Configuration

Define language-specific settings:

```python
# workflows/shared/language/types.py

from typing_extensions import TypedDict


class LanguageConfig(TypedDict):
    """Configuration for a specific language in research workflows.

    Attributes:
        code: ISO 639-1 code (e.g., "es", "zh", "ja")
        name: Full language name (e.g., "Spanish", "Mandarin Chinese")
        search_domains: Preferred domain TLDs (e.g., [".es", ".mx"])
        search_engine_locale: Locale code for search APIs (e.g., "es-ES")
    """
    code: str
    name: str
    search_domains: list[str]
    search_engine_locale: str
```

Register supported languages:

```python
# workflows/shared/language/languages.py

from .types import LanguageConfig

SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        code="en",
        name="English",
        search_domains=[".com", ".org", ".edu"],
        search_engine_locale="en-US",
    ),
    "es": LanguageConfig(
        code="es",
        name="Spanish",
        search_domains=[".es", ".mx", ".ar", ".co"],
        search_engine_locale="es-ES",
    ),
    "zh": LanguageConfig(
        code="zh",
        name="Mandarin Chinese",
        search_domains=[".cn", ".tw", ".hk"],
        search_engine_locale="zh-CN",
    ),
    "ja": LanguageConfig(
        code="ja",
        name="Japanese",
        search_domains=[".jp"],
        search_engine_locale="ja-JP",
    ),
    # ... 29 languages total
}


def get_language_config(code: str) -> LanguageConfig:
    """Get language configuration by ISO 639-1 code."""
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {code}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
    return SUPPORTED_LANGUAGES[code]
```

### Opus Prompt Translation

Translate prompts while preserving structure:

```python
# workflows/shared/language/translator.py

import asyncio
import logging
from cachetools import TTLCache

from workflows.shared.llm_utils import get_llm, ModelTier

logger = logging.getLogger(__name__)

# Cache translated prompts (24h TTL, maxsize=500 for all prompts × languages)
_prompt_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)
_translation_locks: dict[str, asyncio.Lock] = {}

PROMPT_TRANSLATION_SYSTEM = """You are translating an LLM system prompt to another language.

Your task is to produce a high-quality, native-sounding translation that:
1. Preserves the EXACT meaning and instructional intent
2. Uses natural, fluent phrasing (not literal word-for-word)
3. Maintains the same professional, clear, direct tone
4. Keeps the same structure and formatting

CRITICAL RULES:
- Keep all format placeholders EXACTLY as-is: {date}, {query}, {research_brief}, etc.
- Keep JSON schema examples in English - LLMs understand English JSON universally
- Keep technical terms commonly used in English (e.g., "JSON", "API", "URL")
- Keep code examples and variable names in English
- Translate instructional text, section headers, and natural language content

Output ONLY the translated prompt text. No explanations."""


async def translate_prompt(
    english_prompt: str,
    target_language: str,
    cache_key: str | None = None,
) -> str:
    """Translate an English prompt using Opus with caching.

    Uses LLM translation (not machine translation) to preserve semantic intent.

    Args:
        english_prompt: The English prompt to translate
        target_language: Full language name (e.g., "Spanish")
        cache_key: Optional cache key. Format: "{node}_{prompt_type}_{lang}"

    Returns:
        Translated prompt (or original English on failure)
    """
    # Check cache
    if cache_key and cache_key in _prompt_cache:
        logger.debug(f"Prompt cache hit: {cache_key}")
        return _prompt_cache[cache_key]

    # Lock to prevent concurrent translations
    if cache_key:
        if cache_key not in _translation_locks:
            _translation_locks[cache_key] = asyncio.Lock()

        async with _translation_locks[cache_key]:
            if cache_key in _prompt_cache:
                return _prompt_cache[cache_key]

            result = await _do_translation(english_prompt, target_language)
            _prompt_cache[cache_key] = result
            logger.info(f"Cached prompt: {cache_key} ({len(result)} chars)")
            return result
    else:
        return await _do_translation(english_prompt, target_language)


async def _do_translation(english_prompt: str, target_language: str) -> str:
    """Perform translation using Opus."""
    llm = get_llm(ModelTier.OPUS, max_tokens=8192)

    user_prompt = f"""Translate this LLM prompt to {target_language}:

{english_prompt}"""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": PROMPT_TRANSLATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}, falling back to English")
        return english_prompt


async def get_translated_prompt(
    english_prompt: str,
    language_code: str,
    language_name: str,
    prompt_name: str,
) -> str:
    """Convenience function with standard cache key format."""
    if language_code == "en":
        return english_prompt

    cache_key = f"{prompt_name}_{language_code}"
    return await translate_prompt(english_prompt, language_name, cache_key)
```

### Haiku Query Translation

Translate search queries (shorter cache TTL):

```python
# workflows/shared/language/query_translator.py

from cachetools import TTLCache
from workflows.shared.llm_utils import get_llm, ModelTier

# Shorter TTL (1h) for queries since they're shorter and cheaper
_query_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)

QUERY_TRANSLATION_SYSTEM = """Translate this search query to {target_language}.
Preserve search intent. Output ONLY the translated query, nothing else."""


async def translate_query(
    english_query: str,
    target_language: str,
    cache_key: str | None = None,
) -> str:
    """Translate a search query using Haiku."""
    if cache_key and cache_key in _query_cache:
        return _query_cache[cache_key]

    llm = get_llm(ModelTier.HAIKU, max_tokens=200)

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": QUERY_TRANSLATION_SYSTEM.format(target_language=target_language)},
            {"role": "user", "content": english_query},
        ])
        result = response.content.strip()

        if cache_key:
            _query_cache[cache_key] = result

        return result
    except Exception:
        return english_query  # Fallback to English
```

### Workflow Integration

Add language parameter to workflow APIs:

```python
# workflows/research/subgraphs/academic_lit_review/graph/api.py

from workflows.shared.language import get_language_config


async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: str = "standard",
    date_range: tuple[int, int] | None = None,
    language: str = "en",  # NEW: Language parameter
) -> dict:
    """Run academic literature review workflow.

    Args:
        topic: Research topic
        research_questions: List of specific questions
        quality: Quality tier (quick, standard, comprehensive, high_quality)
        date_range: Optional year range filter
        language: Language code (e.g., "en", "es", "zh")
    """
    language_config = get_language_config(language)

    initial_state = {
        "input": LitReviewInput(
            topic=topic,
            research_questions=research_questions,
            quality=quality,
            date_range=date_range,
            language_code=language,  # Stored in input
        ),
        "language_config": language_config,  # Used throughout workflow
        # ...
    }

    return await lit_review_graph.ainvoke(initial_state)
```

Use translated prompts in nodes:

```python
# workflows/research/subgraphs/academic_lit_review/clustering/llm_clustering.py

from workflows.shared.language import get_translated_prompt


async def run_llm_clustering_node(state: ClusteringState) -> dict[str, Any]:
    language_config = state.get("language_config")
    language_code = language_config.get("code", "en") if language_config else "en"
    language_name = language_config.get("name", "English") if language_config else "English"

    # Get translated system prompt
    system_prompt = await get_translated_prompt(
        LLM_CLUSTERING_SYSTEM_PROMPT,
        language_code=language_code,
        language_name=language_name,
        prompt_name="llm_clustering_system",
    )

    llm = get_llm(tier=ModelTier.SONNET, max_tokens=16000)
    structured_llm = llm.with_structured_output(LLMTopicSchemaOutput)

    messages = [
        {"role": "system", "content": system_prompt},  # Translated
        {"role": "user", "content": user_prompt},
    ]

    result = await structured_llm.ainvoke(messages)
    # ...
```

## Usage

```python
from workflows.research.subgraphs.academic_lit_review import academic_lit_review
from workflows.research.subgraphs.book_finding import book_finding

# Academic review in Spanish
result = await academic_lit_review(
    topic="inteligencia artificial en medicina",
    research_questions=["¿Cómo se usa IA para diagnóstico?"],
    quality="standard",
    language="es",
)

# Book finding in Japanese
result = await book_finding(
    theme="組織のレジリエンス",
    quality="standard",
    language="ja",
)
```

## Guidelines

### Cache Strategy

| Content Type | Model | Cache TTL | Cache Size |
|--------------|-------|-----------|------------|
| System prompts | Opus | 24 hours | 500 |
| Search queries | Haiku | 1 hour | 1000 |
| User content | N/A | No cache | N/A |

### Preservation Rules

When translating prompts, always preserve:
1. **Format placeholders**: `{query}`, `{date}`, `{research_brief}`
2. **JSON schemas**: Keep in English (universal)
3. **Technical terms**: API, URL, JSON, DOI, etc.
4. **Code examples**: Variable names, function calls
5. **Citation formats**: APA, MLA, etc.

### Language Selection

Use the appropriate model tier for translation:
- **Opus**: Complex system prompts (preserve tone, structure)
- **Haiku**: Simple search queries (fast, cheap)
- **Sonnet**: User-facing content (balance quality/cost)

### Fallback Strategy

Always fall back to English on translation failure:
```python
except Exception as e:
    logger.warning(f"Translation failed: {e}, using English")
    return english_prompt
```

## Known Uses

- `workflows/research/subgraphs/academic_lit_review/`: `language` parameter
- `workflows/research/subgraphs/book_finding/`: `language` parameter
- `workflows/research/`: Main research workflow multilingual support
- 29 supported languages in `SUPPORTED_LANGUAGES`

## Consequences

### Benefits
- **Native language research**: Search and output in user's language
- **Preserved structure**: Placeholders, JSON, technical terms intact
- **Cost efficiency**: Caching reduces repeated translation costs
- **Graceful degradation**: Falls back to English on failure

### Trade-offs
- **Translation cost**: Opus calls for complex prompts
- **Latency**: First translation adds delay (cached afterward)
- **Quality variation**: Translation quality varies by language pair

## Related Patterns

- [Multi-Source Research Orchestration](../langgraph/multi-source-research-orchestration.md) - Workflow using language config
- [Standalone Book Finding Workflow](../langgraph/standalone-book-finding-workflow.md) - Uses language parameter

## Related Solutions

- [Prompt Caching Patterns](../llm-interaction/prompt-caching-patterns.md) - Caching strategies

## References

- [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
- [cachetools TTLCache](https://cachetools.readthedocs.io/)
