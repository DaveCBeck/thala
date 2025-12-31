"""Pydantic data models for OpenAlex."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OpenAlexAuthor(BaseModel):
    """Author information from OpenAlex."""

    name: str
    institution: Optional[str] = None


class OpenAlexWork(BaseModel):
    """Individual academic work from OpenAlex."""

    title: str
    url: str  # oa_url if available, else DOI (preferred for scraping)
    doi: Optional[str] = None  # Always keep DOI for citations
    oa_url: Optional[str] = None  # Open access URL for full text
    abstract: Optional[str] = None
    authors: list[OpenAlexAuthor] = Field(default_factory=list)
    publication_date: Optional[str] = None
    cited_by_count: int = 0
    primary_topic: Optional[str] = None
    source_name: Optional[str] = None  # Journal/venue name
    is_oa: bool = False  # Whether work is open access
    oa_status: Optional[str] = None  # gold, green, hybrid, bronze, closed


class OpenAlexSearchOutput(BaseModel):
    """Output schema for openalex_search tool."""

    query: str
    total_results: int
    results: list[OpenAlexWork]


class OpenAlexCitationResult(BaseModel):
    """Output schema for citation retrieval."""

    source_doi: str
    direction: Literal["forward", "backward"]
    total_count: int
    results: list[OpenAlexWork]


class OpenAlexAuthorWorksResult(BaseModel):
    """Output schema for author works retrieval."""

    author_id: str
    author_name: str
    total_works: int
    results: list[OpenAlexWork]
