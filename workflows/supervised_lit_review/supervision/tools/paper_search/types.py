"""Types for paper search results."""

from pydantic import BaseModel, Field


class PaperSearchResult(BaseModel):
    """Compact paper metadata for search results."""

    doi: str = Field(description="Paper DOI")
    title: str = Field(description="Paper title")
    year: int = Field(description="Publication year")
    authors: str = Field(description="Authors in 'Smith et al.' format")
    relevance: float = Field(description="Relevance score 0-1")
    zotero_key: str = Field(description="Citation key for [@KEY] format")


class SearchPapersOutput(BaseModel):
    """Output from search_papers tool."""

    query: str = Field(description="The search query")
    total_found: int = Field(description="Number of papers found")
    papers: list[PaperSearchResult] = Field(description="Matching papers")


class PaperContentOutput(BaseModel):
    """Output from get_paper_content tool."""

    doi: str = Field(description="Paper DOI")
    title: str = Field(description="Paper title")
    content: str = Field(description="L2 10:1 compressed content")
    key_findings: list[str] = Field(description="Key findings from extraction")
    truncated: bool = Field(description="Whether content was truncated")
