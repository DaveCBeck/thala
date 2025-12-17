"""Embedding generation service for MCP tools."""

import os
from abc import ABC, abstractmethod

import httpx

from .errors import EmbeddingError


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
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self._client.post(
                "/embeddings",
                json={"input": texts, "model": self.model},
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to ensure correct order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"OpenAI API error: {e.response.status_code} - {e.response.text}",
                provider="openai",
            )
        except Exception as e:
            raise EmbeddingError(str(e), provider="openai")

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
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"Ollama API error: {e.response.status_code}",
                provider="ollama",
            )
        except httpx.ConnectError:
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self.host}. Is Ollama running?",
                provider="ollama",
            )
        except Exception as e:
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
        elif provider == "ollama":
            model = model or os.environ.get("THALA_EMBEDDING_MODEL", "nomic-embed-text")
            host = os.environ.get("THALA_OLLAMA_HOST", "http://localhost:11434")
            self.model = model
            self._provider = OllamaEmbeddings(model=model, host=host)
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

    async def close(self):
        """Close provider resources."""
        await self._provider.close()
