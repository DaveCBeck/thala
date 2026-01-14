"""Pydantic models for Zotero API operations."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ZoteroCreator(BaseModel):
    """Creator (author, editor, etc.) for a Zotero item."""

    firstName: Optional[str] = None
    lastName: Optional[str] = None
    name: Optional[str] = None  # For single-field names
    creatorType: str = "author"


class ZoteroTag(BaseModel):
    """Tag attached to a Zotero item."""

    tag: str
    type: int = 0  # 0 = user tag, 1 = automatic


class ZoteroItem(BaseModel):
    """Full Zotero item returned from the API."""

    key: str = Field(min_length=8, max_length=8, description="8-char Zotero key")
    itemID: int
    itemType: str
    version: int
    libraryID: int
    dateAdded: Optional[str] = None
    dateModified: Optional[str] = None
    fields: dict[str, Any] = Field(default_factory=dict)
    creators: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[dict[str, Any]] = Field(default_factory=list)
    collections: list[int] = Field(default_factory=list)


class ZoteroItemCreate(BaseModel):
    """Schema for creating a new Zotero item."""

    itemType: str = Field(description="Zotero item type (book, journalArticle, etc.)")
    fields: dict[str, Any] = Field(default_factory=dict)
    creators: list[ZoteroCreator] = Field(default_factory=list)
    tags: list[str | ZoteroTag] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)


class ZoteroItemUpdate(BaseModel):
    """Schema for updating a Zotero item."""

    fields: Optional[dict[str, Any]] = None
    creators: Optional[list[ZoteroCreator]] = None
    tags: Optional[list[str | ZoteroTag]] = None
    collections: Optional[list[str]] = None


class ZoteroSearchCondition(BaseModel):
    """Search condition for Zotero search."""

    condition: str  # e.g., "title", "tag", "quicksearch-everything"
    operator: str = "contains"  # e.g., "is", "contains", "doesNotContain"
    value: str = ""
    required: bool = True


class ZoteroSearchResult(BaseModel):
    """Lightweight search result from Zotero."""

    key: str
    itemID: int
    itemType: str
    title: Optional[str] = None
    dateModified: Optional[str] = None


class ZoteroHealthStatus(BaseModel):
    """Health check response from the plugin."""

    healthy: bool
    status: Optional[str] = None
    plugin: Optional[str] = None
    version: Optional[str] = None
    zoteroVersion: Optional[str] = None
    libraryID: Optional[int] = None
    error: Optional[str] = None
