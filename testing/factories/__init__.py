"""Test data factories for generating realistic test objects.

This module provides factory functions for creating test data. Factories
generate valid Pydantic models with sensible defaults that can be overridden.

Usage:
    from testing.factories import make_zotero_item, make_store_record

    # Create with defaults
    item = make_zotero_item()

    # Override specific fields
    book = make_zotero_item(
        item_type="book",
        title="Custom Title",
        creators=[{"firstName": "John", "lastName": "Doe"}],
    )
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from core.stores.schema import (
    BaseRecord,
    CoherenceRecord,
    SourceType,
    StoreRecord,
)
from core.stores.zotero.schemas import (
    ZoteroCreator,
    ZoteroItem,
    ZoteroItemCreate,
)


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# Counter for generating unique test data
_counter = 0


def _next_counter() -> int:
    """Get next counter value for unique test data."""
    global _counter
    _counter += 1
    return _counter


def make_zotero_item_create(
    item_type: str = "journalArticle",
    title: str | None = None,
    creators: list[dict[str, str]] | None = None,
    tags: list[str] | None = None,
    abstract: str | None = None,
    date: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> ZoteroItemCreate:
    """Create a ZoteroItemCreate for testing.

    Args:
        item_type: Zotero item type (journalArticle, book, webpage, etc.)
        title: Item title (auto-generated if not provided)
        creators: List of creator dicts with firstName/lastName or name
        tags: List of tag strings
        abstract: Item abstract
        date: Publication date string
        extra_fields: Additional fields to merge

    Returns:
        ZoteroItemCreate ready for use with ZoteroStore.add()
    """
    n = _next_counter()

    fields: dict[str, Any] = {}

    # Title
    if title is None:
        title = f"Test {item_type.replace('_', ' ').title()} {n}"
    fields["title"] = title

    # Abstract
    if abstract is not None:
        fields["abstractNote"] = abstract
    elif item_type in ("journalArticle", "book", "conferencePaper"):
        fields["abstractNote"] = (
            f"This is the abstract for test item {n}. "
            "It contains a summary of the research findings and methodology."
        )

    # Date
    if date is not None:
        fields["date"] = date
    else:
        fields["date"] = "2024"

    # Type-specific fields
    if item_type == "journalArticle":
        fields.setdefault("publicationTitle", f"Journal of Test Research {n}")
        fields.setdefault("volume", str((n % 10) + 1))
        fields.setdefault("issue", str((n % 4) + 1))
        fields.setdefault("pages", f"{n * 10}-{n * 10 + 15}")
    elif item_type == "book":
        fields.setdefault("publisher", "Test Academic Press")
        fields.setdefault("place", "Cambridge, MA")
        fields.setdefault("ISBN", f"978-0-{n:04d}-{n * 2:04d}-0")
    elif item_type == "webpage":
        fields.setdefault("url", f"https://example.com/test-{n}")
        fields.setdefault("websiteTitle", "Example Research Site")

    # Merge extra fields
    if extra_fields:
        fields.update(extra_fields)

    # Creators
    creator_list = []
    if creators:
        for c in creators:
            creator_list.append(ZoteroCreator(**c))
    else:
        # Default creator
        creator_list.append(
            ZoteroCreator(firstName="Test", lastName=f"Author{n}", creatorType="author")
        )

    return ZoteroItemCreate(
        itemType=item_type,
        fields=fields,
        creators=creator_list,
        tags=tags or [],
    )


def make_zotero_item(
    key: str | None = None,
    item_id: int | None = None,
    item_type: str = "journalArticle",
    title: str | None = None,
    version: int = 1,
    library_id: int = 1,
    fields: dict[str, Any] | None = None,
    creators: list[dict[str, Any]] | None = None,
    tags: list[dict[str, Any]] | None = None,
) -> ZoteroItem:
    """Create a ZoteroItem for testing (as returned from Zotero API).

    Args:
        key: 8-character Zotero key (auto-generated if not provided)
        item_id: Zotero itemID (auto-generated if not provided)
        item_type: Zotero item type
        title: Item title
        version: Item version
        library_id: Library ID
        fields: Item fields dict
        creators: List of creator dicts
        tags: List of tag dicts

    Returns:
        ZoteroItem as returned by ZoteroStore.get()
    """
    n = _next_counter()

    if key is None:
        key = f"TEST{n:04d}"

    if item_id is None:
        item_id = n

    if fields is None:
        fields = {}

    if title is not None:
        fields["title"] = title
    elif "title" not in fields:
        fields["title"] = f"Test {item_type.replace('_', ' ').title()} {n}"

    return ZoteroItem(
        key=key,
        itemID=item_id,
        itemType=item_type,
        version=version,
        libraryID=library_id,
        fields=fields,
        creators=creators or [],
        tags=tags or [],
    )


def make_store_record(
    record_id: UUID | None = None,
    source_type: SourceType = SourceType.EXTERNAL,
    zotero_key: str | None = None,
    content: str | None = None,
    language_code: str = "en",
    compression_level: int = 0,
    metadata: dict[str, Any] | None = None,
) -> StoreRecord:
    """Create a StoreRecord for testing.

    Args:
        record_id: Record UUID (auto-generated if not provided)
        source_type: EXTERNAL or INTERNAL
        zotero_key: 8-char Zotero key (auto-generated for EXTERNAL if not provided)
        content: Text content (auto-generated if not provided)
        language_code: ISO 639-1 language code
        compression_level: 0 for original, 1+ for compressions
        metadata: Additional metadata dict

    Returns:
        StoreRecord for use with stores
    """
    n = _next_counter()

    if record_id is None:
        record_id = uuid4()

    if content is None:
        content = (
            f"This is test content for record {n}. "
            "It contains multiple sentences to simulate real document content. "
            "The content should be substantial enough to generate meaningful embeddings."
        )

    if source_type == SourceType.EXTERNAL and zotero_key is None:
        zotero_key = f"TEST{n:04d}"

    return StoreRecord(
        id=record_id,
        source_type=source_type,
        zotero_key=zotero_key,
        content=content,
        language_code=language_code,
        compression_level=compression_level,
        metadata=metadata or {"test_id": n},
    )


def make_coherence_record(
    record_id: UUID | None = None,
    content: str | None = None,
    confidence: float = 0.8,
    category: str = "belief",
    metadata: dict[str, Any] | None = None,
) -> CoherenceRecord:
    """Create a CoherenceRecord for testing.

    Args:
        record_id: Record UUID (auto-generated if not provided)
        content: Text content (auto-generated if not provided)
        confidence: Confidence score 0.0-1.0
        category: Category (belief, preference, identity, goal)
        metadata: Additional metadata dict

    Returns:
        CoherenceRecord for use with coherence store
    """
    n = _next_counter()

    if record_id is None:
        record_id = uuid4()

    if content is None:
        content = f"Test {category} statement {n} with confidence {confidence}."

    return CoherenceRecord(
        id=record_id,
        content=content,
        confidence=confidence,
        category=category,
        metadata=metadata or {"test_id": n},
    )


def make_base_record(
    record_id: UUID | None = None,
    source_type: SourceType = SourceType.INTERNAL,
    content: str | None = None,
    language_code: str = "en",
    metadata: dict[str, Any] | None = None,
) -> BaseRecord:
    """Create a BaseRecord for testing.

    Args:
        record_id: Record UUID (auto-generated if not provided)
        source_type: EXTERNAL or INTERNAL
        content: Text content (auto-generated if not provided)
        language_code: ISO 639-1 language code
        metadata: Additional metadata dict

    Returns:
        BaseRecord for use with stores
    """
    n = _next_counter()

    if record_id is None:
        record_id = uuid4()

    if content is None:
        content = f"Test base record content {n}."

    return BaseRecord(
        id=record_id,
        source_type=source_type,
        content=content,
        language_code=language_code,
        metadata=metadata or {"test_id": n},
    )


def make_academic_paper_content(
    title: str | None = None,
    sections: list[str] | None = None,
) -> str:
    """Generate realistic academic paper content for testing.

    Args:
        title: Paper title (auto-generated if not provided)
        sections: List of section names (uses defaults if not provided)

    Returns:
        Multi-paragraph academic paper content string
    """
    n = _next_counter()

    if title is None:
        title = f"A Study of Test Phenomena in Domain {n}"

    if sections is None:
        sections = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]

    content_parts = [f"# {title}\n"]

    for section in sections:
        content_parts.append(f"\n## {section}\n")
        if section == "Abstract":
            content_parts.append(
                "This paper presents findings from our investigation into test phenomena. "
                "We employed rigorous methodology and found significant results. "
                "Our conclusions have implications for future research."
            )
        elif section == "Introduction":
            content_parts.append(
                "The study of test phenomena has been a subject of interest for many researchers. "
                "Previous work has established foundational concepts, but gaps remain. "
                "This paper addresses these gaps through novel approaches."
            )
        elif section == "Methods":
            content_parts.append(
                "We utilized a mixed-methods approach combining quantitative analysis with qualitative insights. "
                "Data was collected from multiple sources and validated through cross-referencing. "
                "Statistical analysis was performed using standard techniques."
            )
        elif section == "Results":
            content_parts.append(
                "Our analysis revealed several key findings. "
                "First, we observed significant patterns in the data. "
                "Second, these patterns correlated with theoretical predictions."
            )
        elif section == "Discussion":
            content_parts.append(
                "These results support our initial hypotheses and extend current understanding. "
                "The implications are relevant for both theory and practice. "
                "Limitations of this study include sample size and scope."
            )
        elif section == "Conclusion":
            content_parts.append(
                "In conclusion, this study contributes to the field through novel findings. "
                "Future work should explore related questions. "
                "We recommend continued investigation in this area."
            )
        else:
            content_parts.append(f"Content for {section} section in test paper {n}.")

    return "\n".join(content_parts)


# Export all factory functions
__all__ = [
    "make_zotero_item_create",
    "make_zotero_item",
    "make_store_record",
    "make_coherence_record",
    "make_base_record",
    "make_academic_paper_content",
]
