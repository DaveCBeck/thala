---
module: core/embedding
date: 2026-01-28
problem_type: integration_issue
component: embedding
symptoms:
  - "OpenAI embeddings limited to 8K tokens requiring frequent chunking"
  - "Long documents degraded by chunking and averaging"
  - "Separate token estimation needed for embedding vs generation"
root_cause: config_error
resolution_type: dependency_update
severity: medium
verified_fix: true
tags: [embedding, voyage-ai, openai, migration, token-limits, provider]
---

# Embedding Provider Migration: OpenAI to Voyage AI

## Problem

OpenAI's `text-embedding-3-small` model is limited to 8K tokens (~32K characters), causing frequent chunking for academic documents. Chunking and averaging embeddings degrades semantic representation, especially for documents with distributed themes.

**Symptoms:**
```python
# Before migration (OpenAI 8K limit)
OPENAI_MAX_TOKENS = 8192
SAFE_CHAR_LIMIT = ~32,000 chars

# Academic paper: ~50K chars → 2 chunks → averaged embedding
# Lost: relationships between sections, overall document coherence
```

## Solution

Migrate to Voyage AI embeddings which offer 32K token context (~126K characters), eliminating chunking for most documents while maintaining high-quality embeddings.

### Provider Comparison

| Feature | OpenAI | Voyage AI |
|---------|--------|-----------|
| Model | text-embedding-3-small | voyage-4-large |
| Token Limit | 8,192 | 32,000 |
| Char Limit | ~32K | ~126K |
| Native Async | No (via httpx) | Yes (voyageai.AsyncClient) |
| Auto Retries | Manual | Built-in (max_retries=3) |

### Implementation

**1. Add VoyageAIEmbeddings Provider:**

```python
# core/embedding.py

import voyageai

class VoyageAIEmbeddings(EmbeddingProvider):
    """Voyage AI embeddings provider with native async support."""

    def __init__(
        self,
        model: str = "voyage-4-large",
        api_key: str | None = None,
    ):
        self.model = model
        api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise EmbeddingError(
                "VOYAGE_API_KEY not set.",
                provider="voyage",
            )
        # Native async client with built-in retries
        self._client = voyageai.AsyncClient(
            api_key=api_key,
            max_retries=3,
            timeout=60.0,
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # ... cache checking logic ...

        if uncached_texts:
            response = await self._client.embed(
                texts=uncached_texts,
                model=self.model,
                truncation=True,  # Safety for edge cases
            )
            new_embeddings = response.embeddings

            # Cache results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                set_cached("embeddings", cache_keys[idx], embedding)

        return results
```

**2. Provider-Aware Token Limits:**

```python
# Token limits vary by provider
VOYAGE_MAX_TOKENS = 32000   # Voyage AI models have 32K context
OPENAI_MAX_TOKENS = 8192    # OpenAI text-embedding-3-small
CHARS_PER_TOKEN_ESTIMATE = 4

def get_safe_char_limit(provider: str) -> int:
    """Get safe character limit based on provider's token limit."""
    if provider == "voyage":
        return (VOYAGE_MAX_TOKENS - 500) * CHARS_PER_TOKEN_ESTIMATE  # ~126k chars
    return (OPENAI_MAX_TOKENS - 200) * CHARS_PER_TOKEN_ESTIMATE  # ~32k chars
```

**3. Update Default Provider:**

```python
class EmbeddingService:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ):
        # Default changed from "openai" to "voyage"
        provider = provider or os.environ.get("THALA_EMBEDDING_PROVIDER", "voyage")

        if provider == "voyage":
            model = model or os.environ.get("THALA_EMBEDDING_MODEL", "voyage-4-large")
            self._provider = VoyageAIEmbeddings(model=model)
        elif provider == "openai":
            # Still supported for backward compatibility
            ...
```

**4. Long Text Embedding with Provider Limits:**

```python
async def embed_long(self, text: str) -> list[float]:
    """Generate embedding for potentially long text."""
    safe_limit = get_safe_char_limit(self.provider_name)

    if len(text) <= safe_limit:
        return await self.embed(text)

    # Chunk only if necessary (rare with Voyage)
    chunks = chunk_text_by_sections(text, max_chars=safe_limit)
    embeddings = await self.embed_batch(chunks)
    return np.mean(embeddings, axis=0).tolist()
```

### Environment Configuration

```bash
# .env

# Provider: "voyage" (default), "openai", or "ollama"
THALA_EMBEDDING_PROVIDER=voyage

# Model selection
# Voyage: voyage-4-large (best), voyage-4 (balanced), voyage-4-lite (fast/cheap)
THALA_EMBEDDING_MODEL=voyage-4-large

# Voyage API key (required if using voyage provider)
VOYAGE_API_KEY=pa-...
```

### Migration Path

1. **Get Voyage API key** from https://dash.voyageai.com/
2. **Add to .env**: `VOYAGE_API_KEY=pa-...`
3. **Optional**: Keep `THALA_EMBEDDING_PROVIDER=openai` for gradual migration
4. **Verify**: Test embedding generation works before switching default
5. **Update indexes**: Re-embed if changing model family (embedding dimensions differ)

### Caching Compatibility

Embeddings are cached by model name, so:
- Voyage and OpenAI embeddings have separate cache entries
- Cache keys: `generate_cache_key(model, text)` includes model name
- Migration doesn't invalidate existing cache
- Old OpenAI embeddings still available if switching back

## Files Modified

- `core/embedding.py` - Add VoyageAIEmbeddings class, provider-aware limits
- `.env.example` - Update default provider and add Voyage config
- `requirements.txt` - Add `voyageai` dependency

## Prevention

1. **Use provider-aware limits** - Always call `get_safe_char_limit(provider)` instead of hardcoded values
2. **Prefer higher-context models** - Voyage's 32K tokens handles most documents without chunking
3. **Keep OpenAI as fallback** - Provider can be switched via environment variable

## Related Patterns

- [Long Text Embedding Chunking](./long-text-embedding-chunking.md) - Chunking strategy for oversized documents
- [Hash-Based Persistent Caching](../../patterns/data-pipeline/hash-based-persistent-caching.md) - Embedding cache architecture

## References

- Commit: 192a486853681ee3e426761039c077de1b016378
- [Voyage AI Documentation](https://docs.voyageai.com/)
- [Voyage AI Models](https://docs.voyageai.com/docs/embeddings)
