"""Marker API types and models."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class MarkerJobResult(BaseModel):
    """Result of a completed Marker conversion job."""

    markdown: str
    json_data: Optional[dict[str, Any]] = Field(None, alias="json")
    chunks: Optional[list[dict[str, Any]]] = None
    metadata: dict[str, Any]

    class Config:
        populate_by_name = True
