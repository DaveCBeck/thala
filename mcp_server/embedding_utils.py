"""Embedding utilities for MCP tools."""

from typing import Any

from .errors import EmbeddingError


async def generate_embedding(
    embedding_service: Any | None, content: str
) -> tuple[list[float], str | None]:
    """Generate embedding and return (embedding, model_name)."""
    if not embedding_service:
        return None, None
    try:
        embedding = await embedding_service.embed(content)
        return embedding, embedding_service.model
    except Exception as e:
        raise EmbeddingError(
            str(e), embedding_service.provider_name if embedding_service else None
        )


async def validate_embedding_available(embedding_service: Any | None) -> None:
    """Raise EmbeddingError if embedding service not available."""
    if not embedding_service:
        raise EmbeddingError("Embedding service not configured", None)
