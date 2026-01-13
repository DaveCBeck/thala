"""
Embedding generation service for stores.

Supports OpenAI and Ollama providers.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv

from core.utils import generate_cache_key
from workflows.shared.persistent_cache import get_cached, set_cached

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_CACHE_TTL_DAYS = 90

# OpenAI text-embedding-3-small has 8192 token limit
# Use conservative estimate: ~4 chars per token
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
                    # If single paragraph is too large, split by sentences
                    if len(para) > max_chars:
                        # Just truncate very long paragraphs
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


class EmbeddingError(Exception):
    """Error generating embeddings."""

    def __init__(self, message: str, provider: str | None = None):
        self.message = message
        self.provider = provider
        self.details: dict[str, Any] = {}
        if provider:
            self.details["provider"] = provider
        super().__init__(
            f"Embedding generation failed: {message}. "
            "Check embedding provider configuration and API keys."
        )


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY not set. Set the environment variable or pass api_key.",
                provider="openai",
            )
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        cache_key = generate_cache_key(self.model, text)
        cached = get_cached("embeddings", cache_key, ttl_days=EMBEDDING_CACHE_TTL_DAYS)
        if cached is not None:
            logger.debug(f"Cache hit for embedding (model={self.model})")
            return cached

        logger.debug(f"Cache miss, generating embedding (model={self.model})")
        results = await self.embed_batch([text])
        embedding = results[0]
        set_cached("embeddings", cache_key, embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        cache_keys = [generate_cache_key(self.model, text) for text in texts]
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # Check cache for each text
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            cached = get_cached("embeddings", cache_key, ttl_days=EMBEDDING_CACHE_TTL_DAYS)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.debug(
                f"Generating {len(uncached_texts)}/{len(texts)} embeddings "
                f"({len(texts) - len(uncached_texts)} cached, model={self.model})"
            )
        else:
            logger.debug(f"All {len(texts)} embeddings from cache (model={self.model})")

        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                response = await self._client.post(
                    "/embeddings",
                    json={"input": uncached_texts, "model": self.model},
                )
                response.raise_for_status()
                data = response.json()
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                new_embeddings = [e["embedding"] for e in embeddings]

                # Cache and insert new embeddings
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    results[idx] = embedding
                    set_cached("embeddings", cache_keys[idx], embedding)
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"OpenAI API error: {e.response.status_code} - {e.response.text}"
                )
                raise EmbeddingError(
                    f"OpenAI API error: {e.response.status_code} - {e.response.text}",
                    provider="openai",
                )
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise EmbeddingError(str(e), provider="openai")

        return results

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama local embeddings provider."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host
        self._client = httpx.AsyncClient(base_url=host, timeout=120.0)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        cache_key = generate_cache_key(self.model, text)
        cached = get_cached("embeddings", cache_key, ttl_days=EMBEDDING_CACHE_TTL_DAYS)
        if cached is not None:
            logger.debug(f"Cache hit for embedding (model={self.model})")
            return cached

        logger.debug(f"Cache miss, generating embedding (model={self.model})")
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            set_cached("embeddings", cache_key, embedding)
            return embedding
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code}")
            raise EmbeddingError(
                f"Ollama API error: {e.response.status_code}",
                provider="ollama",
            )
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.host}")
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self.host}. Is Ollama running?",
                provider="ollama",
            )
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingError(str(e), provider="ollama")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (sequential for Ollama)."""
        return [await self.embed(text) for text in texts]

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class EmbeddingService:
    """
    Configurable embedding service.

    Configuration via environment variables:
    - THALA_EMBEDDING_PROVIDER: 'openai' or 'ollama' (default: 'openai')
    - THALA_EMBEDDING_MODEL: model name (default depends on provider)
    - OPENAI_API_KEY: required for OpenAI provider
    - THALA_OLLAMA_HOST: Ollama host (default: http://localhost:11434)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ):
        provider = provider or os.environ.get("THALA_EMBEDDING_PROVIDER", "openai")
        self.provider_name = provider

        if provider == "openai":
            model = model or os.environ.get("THALA_EMBEDDING_MODEL", "text-embedding-3-small")
            self.model = model
            self._provider = OpenAIEmbeddings(model=model)
            logger.info(f"Initialized OpenAI embeddings with model={model}")
        elif provider == "ollama":
            model = model or os.environ.get("THALA_EMBEDDING_MODEL", "nomic-embed-text")
            host = os.environ.get("THALA_OLLAMA_HOST", "http://localhost:11434")
            self.model = model
            self._provider = OllamaEmbeddings(model=model, host=host)
            logger.info(f"Initialized Ollama embeddings with model={model}, host={host}")
        else:
            raise EmbeddingError(
                f"Unknown embedding provider: {provider}. Use 'openai' or 'ollama'."
            )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        return await self._provider.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return await self._provider.embed_batch(texts)

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
            f"Text too long for single embedding ({len(text)} chars, ~{estimate_tokens(text)} tokens). "
            f"Chunking into {len(chunks)} parts and averaging."
        )

        # Embed all chunks
        embeddings = await self.embed_batch(chunks)

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()

        return avg_embedding

    async def close(self):
        """Close provider resources."""
        await self._provider.close()
