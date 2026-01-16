"""Document model with stable references for structural editing.

Provides content-based stable IDs that survive edits, enabling reliable
references to sections and content blocks throughout the editing process.
"""

from dataclasses import dataclass, field
from typing import Literal, Any
import hashlib


@dataclass
class ContentBlock:
    """A paragraph or content unit with stable identity.

    The block_id is derived from the first 200 characters of content,
    making it stable across edits that don't change the block's opening.
    """

    block_id: str
    content: str
    block_type: Literal["paragraph", "list", "code", "quote", "table", "metadata"]
    word_count: int

    @classmethod
    def from_content(
        cls, content: str, block_type: str = "paragraph"
    ) -> "ContentBlock":
        """Create block with content-derived stable ID."""
        content_hash = hashlib.sha256(content[:200].encode()).hexdigest()[:12]
        return cls(
            block_id=f"blk_{content_hash}",
            content=content,
            block_type=block_type,
            word_count=len(content.split()),
        )

    def verify_unchanged(self) -> bool:
        """Check if content still matches the ID hash."""
        expected = hashlib.sha256(self.content[:200].encode()).hexdigest()[:12]
        return self.block_id == f"blk_{expected}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "block_id": self.block_id,
            "content": self.content,
            "block_type": self.block_type,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentBlock":
        """Deserialize from dict."""
        return cls(
            block_id=data["block_id"],
            content=data["content"],
            block_type=data["block_type"],
            word_count=data["word_count"],
        )


@dataclass
class Section:
    """A hierarchical section with stable identity.

    The section_id is derived from the heading and level,
    making it stable as long as the heading doesn't change.
    """

    section_id: str
    heading: str
    level: int  # 1 = H1, 2 = H2, etc.
    blocks: list[ContentBlock] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)
    parent_id: str | None = None

    @classmethod
    def from_heading(
        cls, heading: str, level: int, parent_id: str | None = None
    ) -> "Section":
        """Create section with heading-derived stable ID."""
        heading_hash = hashlib.sha256(f"{level}:{heading}".encode()).hexdigest()[:12]
        return cls(
            section_id=f"sec_{heading_hash}",
            heading=heading,
            level=level,
            parent_id=parent_id,
        )

    @property
    def total_words(self) -> int:
        """Total words in this section including subsections."""
        count = sum(b.word_count for b in self.blocks)
        for sub in self.subsections:
            count += sub.total_words
        return count

    @property
    def block_ids(self) -> list[str]:
        """All block IDs in this section (not subsections)."""
        return [b.block_id for b in self.blocks]

    @property
    def all_block_ids(self) -> list[str]:
        """All block IDs including subsections."""
        ids = [b.block_id for b in self.blocks]
        for sub in self.subsections:
            ids.extend(sub.all_block_ids)
        return ids

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "level": self.level,
            "blocks": [b.to_dict() for b in self.blocks],
            "subsections": [s.to_dict() for s in self.subsections],
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Section":
        """Deserialize from dict."""
        section = cls(
            section_id=data["section_id"],
            heading=data["heading"],
            level=data["level"],
            parent_id=data.get("parent_id"),
        )
        section.blocks = [ContentBlock.from_dict(b) for b in data.get("blocks", [])]
        section.subsections = [
            Section.from_dict(s) for s in data.get("subsections", [])
        ]
        return section


@dataclass
class DocumentModel:
    """Parsed document with stable references and fast lookups.

    Provides O(1) lookup for any section or block by ID, and can
    reconstruct markdown from the model.
    """

    title: str
    sections: list[Section]
    preamble_blocks: list[ContentBlock]  # Content before first section

    # Indexes built on init
    _section_index: dict[str, Section] = field(default_factory=dict, repr=False)
    _block_index: dict[str, tuple[ContentBlock, str]] = field(
        default_factory=dict, repr=False
    )
    # block_id -> (block, parent_section_id or "__preamble__")

    def __post_init__(self):
        self._build_indexes()

    def _build_indexes(self):
        """Build lookup indexes."""

        def index_section(section: Section):
            self._section_index[section.section_id] = section
            for block in section.blocks:
                self._block_index[block.block_id] = (block, section.section_id)
            for sub in section.subsections:
                index_section(sub)

        for block in self.preamble_blocks:
            self._block_index[block.block_id] = (block, "__preamble__")
        for section in self.sections:
            index_section(section)

    def get_section(self, section_id: str) -> Section | None:
        """Get section by ID."""
        return self._section_index.get(section_id)

    def get_block(self, block_id: str) -> ContentBlock | None:
        """Get block by ID."""
        result = self._block_index.get(block_id)
        return result[0] if result else None

    def get_block_context(self, block_id: str) -> tuple[ContentBlock, str] | None:
        """Get block and its parent section ID."""
        return self._block_index.get(block_id)

    def get_section_path(self, section_id: str) -> list[str]:
        """Get hierarchical path like ['Chapter 1', 'Section 1.2']."""
        path = []
        section = self.get_section(section_id)
        while section:
            path.insert(0, section.heading)
            section = (
                self.get_section(section.parent_id) if section.parent_id else None
            )
        return path

    def get_all_sections(self) -> list[Section]:
        """Get all sections in document order (flattened)."""
        sections = []

        def collect(section_list: list[Section]):
            for section in section_list:
                sections.append(section)
                collect(section.subsections)

        collect(self.sections)
        return sections

    def get_section_by_heading(self, heading: str) -> Section | None:
        """Find section by heading text (case-insensitive)."""
        heading_lower = heading.lower()
        for section in self._section_index.values():
            if section.heading.lower() == heading_lower:
                return section
        return None

    @property
    def total_words(self) -> int:
        """Total words in document."""
        count = sum(b.word_count for b in self.preamble_blocks)
        for section in self.sections:
            count += section.total_words
        return count

    @property
    def section_count(self) -> int:
        """Total number of sections including subsections."""

        def count_sections(sections: list[Section]) -> int:
            return sum(1 + count_sections(s.subsections) for s in sections)

        return count_sections(self.sections)

    @property
    def block_count(self) -> int:
        """Total number of content blocks."""
        return len(self._block_index)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for state storage."""
        return {
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "preamble_blocks": [b.to_dict() for b in self.preamble_blocks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentModel":
        """Deserialize from dict."""
        model = cls(
            title=data.get("title", ""),
            sections=[Section.from_dict(s) for s in data.get("sections", [])],
            preamble_blocks=[
                ContentBlock.from_dict(b) for b in data.get("preamble_blocks", [])
            ],
        )
        return model

    def to_markdown(self) -> str:
        """Reconstruct markdown from model."""
        lines = []

        # Title
        if self.title:
            lines.append(f"# {self.title}")
            lines.append("")

        # Preamble
        for block in self.preamble_blocks:
            lines.append(block.content)
            lines.append("")

        # Sections
        def render_section(section: Section):
            prefix = "#" * section.level
            lines.append(f"{prefix} {section.heading}")
            lines.append("")
            for block in section.blocks:
                lines.append(block.content)
                lines.append("")
            for subsection in section.subsections:
                render_section(subsection)

        for section in self.sections:
            render_section(section)

        return "\n".join(lines).strip()

    def render_for_analysis(self) -> str:
        """Render document with section/block IDs for LLM analysis."""
        lines = []

        if self.preamble_blocks:
            lines.append("<preamble>")
            for block in self.preamble_blocks:
                lines.append(f'  <block id="{block.block_id}" type="{block.block_type}">')
                lines.append(f"    {block.content}")
                lines.append("  </block>")
            lines.append("</preamble>")
            lines.append("")

        def render_section(section: Section, indent: int = 0):
            prefix = "  " * indent
            lines.append(
                f'{prefix}<section id="{section.section_id}" '
                f'level="{section.level}" words="{section.total_words}">'
            )
            lines.append(f"{prefix}  <heading>{section.heading}</heading>")

            if section.blocks:
                for block in section.blocks:
                    lines.append(
                        f'{prefix}  <block id="{block.block_id}" '
                        f'type="{block.block_type}" words="{block.word_count}">'
                    )
                    # Truncate very long blocks for analysis
                    content = block.content
                    if len(content) > 2000:
                        content = content[:2000] + "... [truncated]"
                    lines.append(f"{prefix}    {content}")
                    lines.append(f"{prefix}  </block>")

            for sub in section.subsections:
                render_section(sub, indent + 1)

            lines.append(f"{prefix}</section>")

        for section in self.sections:
            render_section(section)

        return "\n".join(lines)

    def get_section_content(self, section_id: str, include_subsections: bool = True) -> str:
        """Get the full content of a section as markdown."""
        section = self.get_section(section_id)
        if not section:
            return ""

        lines = []
        prefix = "#" * section.level
        lines.append(f"{prefix} {section.heading}")
        lines.append("")

        for block in section.blocks:
            lines.append(block.content)
            lines.append("")

        if include_subsections:
            for sub in section.subsections:
                lines.append(self.get_section_content(sub.section_id, True))

        return "\n".join(lines).strip()
