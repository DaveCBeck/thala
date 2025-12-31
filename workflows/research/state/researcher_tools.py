"""Researcher structured output schemas."""

from pydantic import BaseModel, Field


class SearchQueries(BaseModel):
    """Generated search queries for a research question."""

    queries: list[str] = Field(
        description="2-3 specific search queries to find authoritative sources. Each query should be targeted and focus only on the research topic.",
        min_length=1,
        max_length=5,
    )


class QueryValidation(BaseModel):
    """Validation result for a single search query."""

    is_relevant: bool = Field(
        description="Whether the query is relevant to the research question"
    )
    reason: str = Field(
        description="Brief explanation of why the query is or isn't relevant"
    )


class QueryValidationBatch(BaseModel):
    """Batch validation of search queries."""

    validations: list[QueryValidation] = Field(
        description="Validation results for each query in order"
    )
