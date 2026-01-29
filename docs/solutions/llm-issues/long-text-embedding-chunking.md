---
module: embedding
date: 2026-01-03
problem_type: input_validation_error
component: EmbeddingService
symptoms:
  - "openai.BadRequestError: This model's maximum context length is 8192 tokens"
  - "Embedding generation fails for aggregated chapter summaries"
  - "Tenth summary embedding exceeds token limit"
root_cause: token_limit_exceeded
resolution_type: input_chunking
severity: medium
tags: [embeddings, openai, chunking, token-limit, averaging]
---

# Long Text Embedding Chunking

## Problem

Embedding generation failed when text exceeded OpenAI's 8192 token limit:

```
openai.BadRequestError: This model's maximum context length is 8192 tokens,
however you requested 12847 tokens. Please reduce your input.
```

This occurred with "tenth summaries" - aggregated chapter summaries that often exceeded the limit when combining 10-30 chapter summaries into a single document.

## Root Cause

**Aggregated documents can exceed embedding model token limits.**

OpenAI's `text-embedding-3-small` model has an 8192 token limit. The tenth summary workflow aggregates chapter summaries from books, and these aggregated documents frequently exceeded 32k characters (~8k+ tokens).

The original `embed()` method had no length handling:

```python
# PROBLEMATIC: No length check
async def embed(self, text: str) -> list[float]:
    return await self._provider.embed(text)  # Fails if text too long
```

## Solution

**Chunk long text by markdown sections and average the embeddings.**

Added `embed_long()` method that:
1. Checks if text exceeds safe character limit
2. Chunks by markdown sections (## headers), falling back to paragraphs
3. Embeds each chunk separately
4. Averages embeddings using numpy to preserve semantic information

### Implementation

```python
# core/embedding.py

import numpy as np
import re

# OpenAI text-embedding-3-small has 8192 token limit
# Conservative estimate: ~4 chars per token
OPENAI_MAX_TOKENS = 8192
CHARS_PER_TOKEN_ESTIMATE = 4
SAFE_CHAR_LIMIT = (OPENAI_MAX_TOKENS - 200) * CHARS_PER_TOKEN_ESTIMATE  # ~32k chars


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (conservative estimate)."""
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def chunk_text_by_sections(text: str, max_chars: int = SAFE_CHAR_LIMIT) -> list[str]:
    """
    Split text into chunks that fit within token limits.

    Splits on markdown headers (## ) first, then by paragraphs if needed.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []

    # First try splitting by markdown headers
    sections = re.split(r'\n(?=## )', text)

    current_chunk = ""
    for section in sections:
        if not section.strip():
            continue

        # If section itself is too large, split by paragraphs
        if len(section) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Split large section by paragraphs
            paragraphs = section.split('\n\n')
            para_chunk = ""
            for para in paragraphs:
                if len(para_chunk) + len(para) + 2 <= max_chars:
                    para_chunk = para_chunk + "\n\n" + para if para_chunk else para
                else:
                    if para_chunk:
                        chunks.append(para_chunk.strip())
                    # If single paragraph is too large, truncate
                    if len(para) > max_chars:
                        chunks.append(para[:max_chars].strip())
                    else:
                        para_chunk = para
            if para_chunk:
                chunks.append(para_chunk.strip())

        elif len(current_chunk) + len(section) + 2 <= max_chars:
            current_chunk = current_chunk + "\n\n" + section if current_chunk else section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c.strip()]


class EmbeddingService:
    async def embed_long(self, text: str) -> list[float]:
        """
        Generate embedding for potentially long text.

        If text exceeds token limits, chunks it and averages the embeddings.
        This preserves semantic information from all parts of the document.
        """
        if len(text) <= SAFE_CHAR_LIMIT:
            return await self.embed(text)

        # Chunk the text
        chunks = chunk_text_by_sections(text)
        logger.info(
            f"Text too long for single embedding ({len(text)} chars, "
            f"~{estimate_tokens(text)} tokens). "
            f"Chunking into {len(chunks)} parts and averaging."
        )

        # Embed all chunks
        embeddings = await self.embed_batch(chunks)

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        return avg_embedding
```

### Usage

```python
from core.embedding import get_embedding_service

embedding_service = get_embedding_service()

# For potentially long text (aggregated summaries)
embedding = await embedding_service.embed_long(tenth_summary_content)

# For known-short text (single abstracts)
embedding = await embedding_service.embed(abstract)
```

## Chunking Strategy

The chunking uses a hierarchical approach:

1. **Markdown sections first**: Split on `## ` headers (preserves document structure)
2. **Paragraphs second**: If a section is too large, split on `\n\n`
3. **Truncation last resort**: If a single paragraph exceeds limit, truncate

This preserves semantic coherence better than arbitrary character splitting.

## Why Averaging Works

Averaging embeddings preserves semantic information because:
- Embeddings are normalized vectors in semantic space
- Related chunks will have similar directions
- The average points toward the semantic "center" of the document
- Information from all chunks contributes to the final embedding

Alternative approaches considered:
- **First chunk only**: Loses information from later content
- **Concatenation with truncation**: Arbitrarily cuts content
- **Weighted averaging**: More complex, marginal benefit

## Files Modified

- `core/embedding.py` - Added `embed_long()`, `chunk_text_by_sections()`, `estimate_tokens()`
- `workflows/research/synthesis/nodes/save_tenth_summary.py` - Use `embed_long()` for tenth summaries

## Prevention

When storing embeddings for potentially long content:
1. Use `embed_long()` instead of `embed()` for aggregated content
2. Check content length before embedding if using `embed()` directly
3. Consider chunking at the source (e.g., summarize chapters shorter)

## Testing

```python
async def test_embed_long_handles_large_text():
    embedding_service = get_embedding_service()

    # Generate text exceeding limit
    long_text = "## Section 1\n" + "A" * 50000 + "\n## Section 2\n" + "B" * 50000

    # Should not raise
    embedding = await embedding_service.embed_long(long_text)

    # Should return valid embedding
    assert len(embedding) == 1536  # text-embedding-3-small dimension
    assert all(isinstance(x, float) for x in embedding)


async def test_chunk_text_by_sections():
    text = "## Intro\nShort intro\n\n## Chapter 1\n" + "A" * 40000
    chunks = chunk_text_by_sections(text, max_chars=32000)

    assert len(chunks) == 2  # Should split
    assert all(len(c) <= 32000 for c in chunks)
```

## Related Patterns

- [Hash-Based Persistent Caching](../../patterns/data-pipeline/hash-based-persistent-caching.md) - Embedding cache uses text hash

## References

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [text-embedding-3-small limits](https://platform.openai.com/docs/models/embeddings)
