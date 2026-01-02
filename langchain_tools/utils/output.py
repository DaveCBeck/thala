"""Output utilities for LangChain tools."""

from pydantic import BaseModel


def output_dict(model: BaseModel) -> dict:
    """Convert Pydantic model to JSON-compatible dict for tool returns."""
    return model.model_dump(mode="json")
