"""Document model with stable references for structural editing.

Provides content-based stable IDs that survive edits, enabling reliable
references to sections and content blocks throughout the editing process.

Features:
- Content-based stable IDs for blocks and sections
- Hierarchical anchoring (e.g., "sec_123/block_2" paths)
- Transaction support for safe edit operations
"""

from dataclasses import dataclass, field
from typing import Literal, Any, Generator
from contextlib import contextmanager
import copy
import hashlib
import re


def _normalize_heading(text: str) -> str:
    """Normalize heading for comparison.

    Strips:
    - Leading section numbers (e.g., "1.", "1.2.", "3.2.1")
    - Leading "Chapter/Section X" prefixes
    - All punctuation and whitespace
    - Converts to lowercase

    Examples:
        "1. Introduction" → "introduction"
        "Chapter 3: Methods" → "methods"
        "## 1.2 Results" → "results"
    """
    text = text.lower()
    # Strip leading section/chapter numbers: "1.", "1.2.", "Chapter 1:", "Section 2.3"
    text = re.sub(r'^(?:chapter|section)?\s*[\d.]+[.:)]*\s*', '', text)
    # Remove remaining non-alphabetic characters
    return re.sub(r'[^a-z]', '', text)


def _strip_leading_header(content: str, section_heading: str) -> str:
    """Strip leading markdown header if it matches section heading.

    When LLM-generated content includes a header that duplicates the section's
    own heading, this removes the duplicate to avoid double headers in output.
    """
    # Match markdown header at start of content
    header_match = re.match(r'^(#{1,6})\s+(.+?)(?:\n|$)', content.strip())
    if not header_match:
        return content

    header_text = header_match.group(2).strip()

    # Compare normalized headings
    if _normalize_heading(header_text) == _normalize_heading(section_heading):
        # Strip the duplicate header line
        stripped = content.strip()[header_match.end():].lstrip('\n')
        return stripped

    return content


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

    def get_block_anchor(self, block_id: str) -> str | None:
        """Get hierarchical anchor for a block within this section.

        Returns path like "section_id/block_id" or None if not found.
        """
        for block in self.blocks:
            if block.block_id == block_id:
                return f"{self.section_id}/{block_id}"
        for sub in self.subsections:
            anchor = sub.get_block_anchor(block_id)
            if anchor:
                return f"{self.section_id}/{anchor}"
        return None

    def get_subsection_anchor(self, section_id: str) -> str | None:
        """Get hierarchical anchor for a subsection.

        Returns path like "parent_id/child_id" or None if not found.
        """
        for sub in self.subsections:
            if sub.section_id == section_id:
                return f"{self.section_id}/{section_id}"
            # Check nested subsections
            nested = sub.get_subsection_anchor(section_id)
            if nested:
                return f"{self.section_id}/{nested}"
        return None

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

        # Preamble - skip blocks that are just the title header
        for block in self.preamble_blocks:
            content = block.content.strip()
            # Skip if this block is just a header matching the title
            if self.title:
                content_stripped = _strip_leading_header(content, self.title)
                if not content_stripped.strip():
                    continue  # Block was only a duplicate title header
                content = content_stripped
            if content:
                lines.append(content)
                lines.append("")

        # Sections
        def render_section(section: Section):
            prefix = "#" * section.level
            lines.append(f"{prefix} {section.heading}")
            lines.append("")
            for i, block in enumerate(section.blocks):
                content = block.content
                # Strip leading header from first block if it duplicates section heading
                if i == 0:
                    content = _strip_leading_header(content, section.heading)
                if content.strip():  # Only add non-empty content
                    lines.append(content)
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

    # =========================================================================
    # Hierarchical Anchoring
    # =========================================================================

    def get_anchor(self, element_id: str) -> str | None:
        """Get hierarchical anchor path for any section or block.

        Returns paths like:
        - "sec_abc123" for top-level sections
        - "sec_abc123/sec_def456" for nested sections
        - "sec_abc123/blk_xyz789" for blocks
        - "__preamble__/blk_xyz789" for preamble blocks
        """
        # Check preamble
        for block in self.preamble_blocks:
            if block.block_id == element_id:
                return f"__preamble__/{element_id}"

        # Check sections
        for section in self.sections:
            if section.section_id == element_id:
                return element_id

            # Check blocks in this section
            anchor = section.get_block_anchor(element_id)
            if anchor:
                return anchor

            # Check subsections
            sub_anchor = section.get_subsection_anchor(element_id)
            if sub_anchor:
                return sub_anchor

        return None

    def resolve_anchor(self, anchor: str) -> Section | ContentBlock | None:
        """Resolve a hierarchical anchor to its element.

        Accepts paths like "sec_abc123/blk_xyz789" and returns the element.
        """
        parts = anchor.split("/")
        if not parts:
            return None

        if parts[0] == "__preamble__":
            if len(parts) == 2:
                return self.get_block(parts[1])
            return None

        # Navigate through the path
        current_section = self.get_section(parts[0])
        if not current_section:
            return None

        # If just a section ID, return the section
        if len(parts) == 1:
            return current_section

        # Navigate through remaining parts
        for i, part in enumerate(parts[1:], 1):
            # Check if it's a block
            if part.startswith("blk_"):
                for block in current_section.blocks:
                    if block.block_id == part:
                        return block
                return None

            # Check if it's a subsection
            found = False
            for sub in current_section.subsections:
                if sub.section_id == part:
                    current_section = sub
                    found = True
                    break

            if not found:
                return None

        return current_section

    def get_insertion_point(self, anchor: str, position: str = "after") -> dict | None:
        """Get insertion point metadata for placing content relative to an anchor.

        Args:
            anchor: Hierarchical anchor path
            position: "before", "after", "start" (beginning of section), "end" (end of section)

        Returns:
            Dict with insertion metadata or None if anchor not found
        """
        element = self.resolve_anchor(anchor)
        if not element:
            return None

        if isinstance(element, Section):
            return {
                "type": "section",
                "section_id": element.section_id,
                "position": position,
                "parent_id": element.parent_id,
            }
        elif isinstance(element, ContentBlock):
            # Find the containing section
            context = self.get_block_context(element.block_id)
            if context:
                block, section_id = context
                section = self.get_section(section_id) if section_id != "__preamble__" else None
                block_idx = None
                if section:
                    for i, b in enumerate(section.blocks):
                        if b.block_id == element.block_id:
                            block_idx = i
                            break
                return {
                    "type": "block",
                    "block_id": element.block_id,
                    "section_id": section_id,
                    "block_index": block_idx,
                    "position": position,
                }
        return None

    # =========================================================================
    # Semantic Deduplication
    # =========================================================================

    def deduplicate_sections(self) -> list[str]:
        """Remove duplicate sections with similar headings, keeping the best version.

        Finds sections with normalized-matching headings (e.g., multiple "Conclusion"
        sections) and keeps only the one with the most content/citations.

        Returns list of removed section descriptions for logging.
        """
        removed = []

        # Group top-level sections by normalized heading
        heading_groups: dict[str, list[tuple[int, Section]]] = {}
        for i, section in enumerate(self.sections):
            norm = _normalize_heading(section.heading)
            if norm not in heading_groups:
                heading_groups[norm] = []
            heading_groups[norm].append((i, section))

        # Find groups with duplicates
        indices_to_remove = []
        for norm_heading, group in heading_groups.items():
            if len(group) <= 1:
                continue

            # Score each duplicate: prefer more words, more citations
            def score_section(sec: Section) -> tuple[int, int]:
                words = sec.total_words
                citations = sum(
                    block.content.count('[@') + block.content.count('](')
                    for block in sec.blocks
                )
                return (words, citations)

            # Sort by score descending, keep the best
            scored = [(score_section(sec), i, sec) for i, sec in group]
            scored.sort(key=lambda x: x[0], reverse=True)

            # Keep first (best), mark rest for removal
            for (score, idx, sec) in scored[1:]:
                indices_to_remove.append(idx)
                removed.append(
                    f"Removed duplicate '{sec.heading}' "
                    f"({sec.total_words} words, score={score})"
                )

        # Remove in reverse order to preserve indices
        for idx in sorted(indices_to_remove, reverse=True):
            del self.sections[idx]

        # Also deduplicate within preamble - remove duplicate content blocks
        seen_content_hashes = set()
        new_preamble = []
        for block in self.preamble_blocks:
            content_hash = hashlib.md5(block.content[:200].encode()).hexdigest()
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                new_preamble.append(block)
            else:
                removed.append(f"Removed duplicate preamble block: {block.content[:50]}...")
        self.preamble_blocks = new_preamble

        # Rebuild indexes after modifications
        if removed:
            self._build_indexes()

        return removed

    # =========================================================================
    # Transaction Support
    # =========================================================================

    @contextmanager
    def transaction(self) -> Generator["DocumentTransaction", None, None]:
        """Create a transaction for safe edit operations.

        Usage:
            with document_model.transaction() as txn:
                txn.insert_block_after(section_id, block)
                txn.move_section(source_id, target_id, position)
                if not txn.verify():
                    txn.rollback()
                # Auto-commits on successful exit
        """
        txn = DocumentTransaction(self)
        try:
            yield txn
            if not txn._rolled_back:
                txn.commit()
        except Exception:
            txn.rollback()
            raise


@dataclass
class DocumentTransaction:
    """Transaction wrapper for safe document edits.

    Allows atomic edit operations with rollback capability.
    """

    _original_model: DocumentModel
    _working_copy: DocumentModel = field(init=False)
    _operations: list[dict] = field(default_factory=list)
    _committed: bool = False
    _rolled_back: bool = False

    def __post_init__(self):
        # Create a working copy for modifications
        self._working_copy = DocumentModel.from_dict(
            copy.deepcopy(self._original_model.to_dict())
        )

    def insert_section_after(
        self, after_section_id: str, new_section: Section
    ) -> bool:
        """Insert a new section after the specified section.

        Returns True if successful.
        """
        # Find the section to insert after
        for i, section in enumerate(self._working_copy.sections):
            if section.section_id == after_section_id:
                self._working_copy.sections.insert(i + 1, new_section)
                self._operations.append({
                    "type": "insert_section",
                    "after": after_section_id,
                    "section_id": new_section.section_id,
                })
                self._working_copy._build_indexes()
                return True
            # Check subsections recursively
            # (For simplicity, only handle top-level insertion for now)

        return False

    def insert_block_at_end(
        self, section_id: str, block: ContentBlock
    ) -> bool:
        """Insert a block at the end of a section."""
        section = self._working_copy.get_section(section_id)
        if section:
            section.blocks.append(block)
            self._operations.append({
                "type": "insert_block",
                "section_id": section_id,
                "block_id": block.block_id,
                "position": "end",
            })
            self._working_copy._build_indexes()
            return True
        return False

    def insert_block_at_start(
        self, section_id: str, block: ContentBlock
    ) -> bool:
        """Insert a block at the start of a section."""
        section = self._working_copy.get_section(section_id)
        if section:
            section.blocks.insert(0, block)
            self._operations.append({
                "type": "insert_block",
                "section_id": section_id,
                "block_id": block.block_id,
                "position": "start",
            })
            self._working_copy._build_indexes()
            return True
        return False

    def delete_section(self, section_id: str) -> bool:
        """Delete a section by ID."""
        for i, section in enumerate(self._working_copy.sections):
            if section.section_id == section_id:
                del self._working_copy.sections[i]
                self._operations.append({
                    "type": "delete_section",
                    "section_id": section_id,
                })
                self._working_copy._build_indexes()
                return True
        return False

    def delete_block(self, block_id: str) -> bool:
        """Delete a block by ID."""
        context = self._working_copy.get_block_context(block_id)
        if not context:
            return False

        block, section_id = context
        if section_id == "__preamble__":
            for i, b in enumerate(self._working_copy.preamble_blocks):
                if b.block_id == block_id:
                    del self._working_copy.preamble_blocks[i]
                    self._operations.append({
                        "type": "delete_block",
                        "block_id": block_id,
                    })
                    self._working_copy._build_indexes()
                    return True
        else:
            section = self._working_copy.get_section(section_id)
            if section:
                for i, b in enumerate(section.blocks):
                    if b.block_id == block_id:
                        del section.blocks[i]
                        self._operations.append({
                            "type": "delete_block",
                            "block_id": block_id,
                        })
                        self._working_copy._build_indexes()
                        return True
        return False

    def verify(self) -> dict:
        """Verify the transaction produces valid results.

        Returns dict with:
        - valid: bool indicating if document is valid
        - issues: list of issues found
        """
        issues = []

        # Check for content in reference sections that shouldn't be there
        for section in self._working_copy.get_all_sections():
            heading_lower = section.heading.lower()
            if any(kw in heading_lower for kw in ["reference", "bibliography"]):
                # Reference sections should only have citation-like content
                for block in section.blocks:
                    # Check if block looks like non-citation content
                    if len(block.content) > 500 and not block.content.strip().startswith("["):
                        issues.append(
                            f"Large non-citation content in '{section.heading}'"
                        )

        # Check for orphaned sections (no content)
        for section in self._working_copy.get_all_sections():
            if not section.blocks and not section.subsections:
                if section.heading.lower() not in ["references", "bibliography"]:
                    issues.append(f"Empty section: '{section.heading}'")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def commit(self) -> DocumentModel:
        """Commit the transaction, returning the modified model."""
        if self._rolled_back:
            raise RuntimeError("Cannot commit a rolled-back transaction")
        if self._committed:
            raise RuntimeError("Transaction already committed")

        self._committed = True
        # Update the original model's data
        self._original_model.sections = self._working_copy.sections
        self._original_model.preamble_blocks = self._working_copy.preamble_blocks
        self._original_model._build_indexes()
        return self._original_model

    def rollback(self) -> None:
        """Rollback the transaction, discarding all changes."""
        self._rolled_back = True
        self._operations.clear()
        # Working copy is discarded; original remains unchanged

    def get_result(self) -> DocumentModel:
        """Get the working copy of the document (for inspection before commit)."""
        return self._working_copy

    @property
    def operations(self) -> list[dict]:
        """Get the list of operations performed in this transaction."""
        return self._operations.copy()
