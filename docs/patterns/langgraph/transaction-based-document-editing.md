---
name: transaction-based-document-editing
title: "Transaction-Based Document Editing: Safe Atomic Operations with Rollback"
date: 2026-01-28
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/9b707937fe8a3347262be621eb313ae2
article_path: .context/libs/thala-dev/content/2026-01-28-transaction-based-document-editing-langgraph.md
applicability:
  - "Document editing workflows with LLM-generated content"
  - "Multi-edit operations requiring atomic commit/rollback"
  - "Workflows needing verification before applying changes"
  - "Nested document structures requiring stable references"
components: [document_transaction, hierarchical_anchoring, deep_copy, header_stripping, verification]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [transactions, rollback, document-model, editing, state-mutation, atomic-operations, hierarchical-paths]
---

# Transaction-Based Document Editing: Safe Atomic Operations with Rollback

## Intent

Provide a transaction API for document editing that enables atomic commit/rollback semantics, hierarchical content referencing via stable anchors, and automatic cleanup of LLM output artifacts like duplicate headers.

## Motivation

LLM-generated document edits are inherently unreliable:

**The Problem:**
```python
# UNSAFE: Direct mutation without rollback
document_model = DocumentModel.from_dict(state["document_model"])

for edit in planned_edits:
    # What if edit 3 of 5 fails validation?
    # Earlier edits already mutated the model!
    section = document_model.get_section(edit["target"])
    section.blocks.append(new_block)  # Direct mutation

# Now document_model is in invalid partial state
# Cannot recover original state
# No verification before commit
```

**Additional Issues:**
1. LLMs generate headers even when told not to (duplicate headers)
2. Content placed in wrong sections (e.g., body text in References)
3. Nested block references change during editing
4. Parallel edits can interfere without isolation

**The Solution:**
```python
# SAFE: Transaction with rollback
with document_model.transaction() as txn:
    for edit in planned_edits:
        txn.insert_block_at_end(edit["target"], new_block)

    verification = txn.verify()
    if not verification["valid"]:
        print(f"Issues: {verification['issues']}")
        txn.rollback()
        # Original document_model unchanged!

    # Auto-commits on successful exit
```

## Applicability

Use this pattern when:
- Applying multiple edits that should succeed or fail together
- Need to verify document validity before committing changes
- Working with LLM-generated content that may include unwanted artifacts
- Requiring stable references to nested content during editing
- Multiple parallel edit operations need isolation

Do NOT use this pattern when:
- Single, simple edit that cannot fail
- Performance-critical path where deep copy is too expensive
- Document model is immutable by design
- No verification or rollback needed

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  DocumentModel                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Original State (protected)                                   │ │
│  │  - sections: [Section, ...]                                   │ │
│  │  - preamble_blocks: [ContentBlock, ...]                      │ │
│  │  - _section_index: {id: Section}                             │ │
│  │  - _block_index: {id: (block, section_id)}                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                     │
│                   document_model.transaction()                     │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  DocumentTransaction (working copy)                          │ │
│  │                                                               │ │
│  │  _working_copy = deepcopy(original)                          │ │
│  │  _operations: []  # Log of changes                           │ │
│  │                                                               │ │
│  │  Methods:                                                     │ │
│  │  - insert_section_after(after_id, section) -> bool           │ │
│  │  - insert_block_at_end(section_id, block) -> bool           │ │
│  │  - insert_block_at_start(section_id, block) -> bool         │ │
│  │  - delete_section(section_id) -> bool                        │ │
│  │  - delete_block(block_id) -> bool                           │ │
│  │  - verify() -> {"valid": bool, "issues": [...]}              │ │
│  │  - commit() -> DocumentModel                                 │ │
│  │  - rollback() -> None                                        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  Hierarchical Anchors:                                            │
│  - "sec_abc123"                    # Top-level section            │
│  - "sec_abc123/sec_def456"         # Nested section               │
│  - "sec_abc123/blk_xyz789"         # Block in section             │
│  - "__preamble__/blk_xyz789"       # Preamble block               │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Create Transaction Context Manager

```python
# workflows/enhance/editing/document_model.py

from contextlib import contextmanager
from dataclasses import dataclass, field
import copy
from typing import Generator


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
        """Insert a new section after the specified section."""
        if self._committed or self._rolled_back:
            return False

        # Find insertion point
        for i, section in enumerate(self._working_copy.sections):
            if section.section_id == after_section_id:
                self._working_copy.sections.insert(i + 1, new_section)
                self._working_copy._build_indexes()
                self._operations.append({
                    "type": "insert_section_after",
                    "after_id": after_section_id,
                    "section_id": new_section.section_id,
                })
                return True
        return False

    def insert_block_at_end(self, section_id: str, block: ContentBlock) -> bool:
        """Append block to end of section."""
        if self._committed or self._rolled_back:
            return False

        section = self._working_copy.get_section(section_id)
        if section:
            section.blocks.append(block)
            self._working_copy._build_indexes()
            self._operations.append({
                "type": "insert_block_at_end",
                "section_id": section_id,
                "block_id": block.block_id,
            })
            return True
        return False

    def verify(self) -> dict:
        """Validate document integrity before commit."""
        issues = []

        # Check for content in References section
        for section in self._working_copy.sections:
            if section.heading.lower() in ("references", "bibliography"):
                if any(b.content.strip() for b in section.blocks):
                    issues.append(f"Content found in {section.heading} section")

        # Check for orphaned sections (no blocks, no subsections)
        for section in self._working_copy.get_leaf_sections():
            if not section.blocks:
                issues.append(f"Empty section: {section.heading}")

        return {"valid": len(issues) == 0, "issues": issues}

    def commit(self) -> DocumentModel:
        """Apply all changes to original model."""
        if self._committed:
            raise RuntimeError("Transaction already committed")
        if self._rolled_back:
            raise RuntimeError("Transaction was rolled back")

        # Copy working state to original
        self._original_model.sections = self._working_copy.sections
        self._original_model.preamble_blocks = self._working_copy.preamble_blocks
        self._original_model._build_indexes()
        self._committed = True
        return self._original_model

    def rollback(self) -> None:
        """Discard all changes."""
        self._working_copy = None
        self._operations.clear()
        self._rolled_back = True


class DocumentModel:
    """Document model with transaction support."""

    @contextmanager
    def transaction(self) -> Generator[DocumentTransaction, None, None]:
        """Create a transaction for safe edit operations."""
        txn = DocumentTransaction(self)
        try:
            yield txn
            if not txn._rolled_back:
                txn.commit()
        except Exception:
            txn.rollback()
            raise
```

### Step 2: Implement Hierarchical Anchoring

```python
# workflows/enhance/editing/document_model.py

class DocumentModel:
    """Document model with hierarchical anchor support."""

    def get_anchor(self, element_id: str) -> str | None:
        """Get hierarchical anchor path for any section or block.

        Returns paths like:
        - "sec_abc123" for top-level sections
        - "sec_abc123/sec_def456" for nested sections
        - "sec_abc123/blk_xyz789" for blocks
        - "__preamble__/blk_xyz789" for preamble blocks
        """
        # Check preamble blocks
        for block in self.preamble_blocks:
            if block.block_id == element_id:
                return f"__preamble__/{element_id}"

        # Check sections and their contents
        for section in self.sections:
            anchor = self._get_anchor_recursive(section, element_id)
            if anchor:
                return anchor

        return None

    def _get_anchor_recursive(
        self, section: Section, element_id: str, path: str = ""
    ) -> str | None:
        """Recursively search for element and build anchor path."""
        current_path = f"{path}/{section.section_id}" if path else section.section_id

        # Check if this is the target section
        if section.section_id == element_id:
            return current_path

        # Check blocks in this section
        for block in section.blocks:
            if block.block_id == element_id:
                return f"{current_path}/{element_id}"

        # Check subsections
        for sub in section.subsections:
            anchor = self._get_anchor_recursive(sub, element_id, current_path)
            if anchor:
                return anchor

        return None

    def resolve_anchor(self, anchor: str) -> Section | ContentBlock | None:
        """Resolve a hierarchical anchor to its element."""
        parts = anchor.split("/")
        if not parts:
            return None

        # Handle preamble
        if parts[0] == "__preamble__":
            if len(parts) == 2:
                return self.get_block(parts[1])
            return None

        # Navigate through the path
        current_section = self.get_section(parts[0])
        if not current_section:
            return None

        if len(parts) == 1:
            return current_section

        # Navigate remaining parts
        for part in parts[1:]:
            if part.startswith("blk_"):
                for block in current_section.blocks:
                    if block.block_id == part:
                        return block
                return None

            # Find subsection
            found = False
            for sub in current_section.subsections:
                if sub.section_id == part:
                    current_section = sub
                    found = True
                    break

            if not found:
                return None

        return current_section
```

### Step 3: Add Duplicate Header Stripping

```python
# workflows/enhance/editing/document_model.py

import re


def _normalize_heading(text: str) -> str:
    """Normalize heading for comparison.

    Strips leading section numbers, punctuation, converts to lowercase.
    """
    text = text.lower()
    # Strip "1.", "1.2.", "Chapter 1:", "Section 2.3"
    text = re.sub(r'^(?:chapter|section)?\s*[\d.]+[.:)]*\s*', '', text)
    # Remove non-alphabetic characters
    return re.sub(r'[^a-z]', '', text)


def _strip_leading_header(content: str, section_heading: str | None = None) -> str:
    """Strip leading markdown header from content.

    When LLM-generated content includes a header (despite instructions not to),
    this removes it to avoid breaking document structure.
    """
    header_match = re.match(r'^(#{1,6})\s+(.+?)(?:\n|$)', content.strip())
    if not header_match:
        return content

    header_text = header_match.group(2).strip()

    # If no section_heading provided, strip any header unconditionally
    if section_heading is None:
        return content.strip()[header_match.end():].lstrip('\n')

    # Only strip if it matches the section heading
    if _normalize_heading(header_text) == _normalize_heading(section_heading):
        return content.strip()[header_match.end():].lstrip('\n')

    return content


class DocumentModel:
    def to_markdown(self) -> str:
        """Reconstruct markdown from model with header cleanup."""
        lines = []

        # Preamble with duplicate header stripping
        for block in self.preamble_blocks:
            content = block.content.strip()
            if self.title:
                content_stripped = _strip_leading_header(content, self.title)
                if not content_stripped.strip():
                    continue  # Block was only a duplicate title header
                content = content_stripped
            if content:
                lines.append(content)
                lines.append("")

        # Sections with header stripping
        def render_section(section: Section):
            prefix = "#" * section.level
            lines.append(f"{prefix} {section.heading}")
            lines.append("")
            for block in section.blocks:
                content = _strip_leading_header(block.content, section.heading)
                if content.strip():
                    lines.append(content)
                    lines.append("")
            for subsection in section.subsections:
                render_section(subsection)

        for section in self.sections:
            render_section(section)

        return "\n".join(lines).strip()
```

### Step 4: Use Deep Copy in Edit Assembly

```python
# workflows/enhance/editing/nodes/execute_edits.py

import copy
from langsmith import traceable


@traceable(run_type="chain", name="EditingAssembleEdits")
async def assemble_edits_node(state: dict) -> dict[str, Any]:
    """Assemble completed edits into updated document model."""
    # CRITICAL: Use deep copy to avoid mutating original state
    source_model_data = state.get("updated_document_model", state["document_model"])
    document_model = DocumentModel.from_dict(copy.deepcopy(source_model_data))

    completed_edits = state.get("completed_edits", [])
    if not completed_edits:
        return {
            "updated_document_model": state["document_model"],
            "execution_complete": True,
        }

    successful_edits = [e for e in completed_edits if e.get("success")]
    placement_issues = []

    # Deep copy sections to avoid mutation during assembly
    new_sections = copy.deepcopy(document_model.sections)
    new_preamble = copy.deepcopy(document_model.preamble_blocks)

    for edit in successful_edits:
        # Apply edit to new_sections (not original)
        target_section = _find_section(new_sections, edit["target_id"])
        if target_section:
            new_block = ContentBlock.from_content(edit["content"], "paragraph")
            target_section.blocks.append(new_block)
        else:
            placement_issues.append(f"Target not found: {edit['target_id']}")

    # Create updated model from deep-copied sections
    updated_model = DocumentModel(
        title=document_model.title,
        sections=new_sections,
        preamble_blocks=new_preamble,
    )

    return {
        "updated_document_model": updated_model.to_dict(),
        "execution_complete": True,
        "placement_issues": placement_issues,
    }
```

## Complete Example

```python
from workflows.enhance.editing.document_model import DocumentModel, Section, ContentBlock

# Load document
document_model = DocumentModel.from_dict(state["document_model"])

# Use transaction for safe multi-edit operation
with document_model.transaction() as txn:
    # Add synthesis section after methodology
    synthesis_section = Section.from_heading("Synthesis", level=2)
    synthesis_block = ContentBlock.from_content(
        "This section synthesizes findings from the reviewed literature...",
        "paragraph"
    )
    synthesis_section.blocks.append(synthesis_block)

    txn.insert_section_after("sec_methodology_abc123", synthesis_section)

    # Add content to existing section
    additional_content = ContentBlock.from_content(
        "Recent studies have further confirmed these findings...",
        "paragraph"
    )
    txn.insert_block_at_end("sec_discussion_def456", additional_content)

    # Verify before commit
    verification = txn.verify()
    if not verification["valid"]:
        logger.warning(f"Verification failed: {verification['issues']}")
        txn.rollback()
        # Original document_model remains unchanged
    # Otherwise auto-commits on exit

# Using hierarchical anchors
block_anchor = document_model.get_anchor("blk_content_xyz789")
# Returns: "sec_results_abc123/blk_content_xyz789"

element = document_model.resolve_anchor("sec_results_abc123/blk_content_xyz789")
# Returns: ContentBlock object

# Render with automatic header cleanup
markdown = document_model.to_markdown()
# Duplicate headers from LLM content automatically stripped
```

## Consequences

### Benefits

- **Atomic operations**: All edits succeed or fail together
- **Safe rollback**: Original document preserved until explicit commit
- **Pre-commit verification**: Detect invalid states before applying
- **Stable references**: Hierarchical anchors survive structure changes
- **LLM artifact cleanup**: Automatic duplicate header removal
- **State isolation**: Parallel edits don't interfere
- **Audit trail**: Operation log tracks all changes

### Trade-offs

- **Memory overhead**: Deep copy doubles memory during transaction
- **Performance cost**: Copy and index rebuild add latency
- **Complexity**: More code than direct mutation
- **Learning curve**: Transaction API differs from direct manipulation

### Alternatives

- **Event sourcing**: Store events instead of snapshots (heavier weight)
- **Soft versioning**: Keep version history without full transaction semantics
- **Direct mutation with backup**: Copy before edit, restore on failure (less safe)

## Related Patterns

- [Multi-Phase Document Editing](./multi-phase-document-editing.md) - Four-phase editing with verification
- [Section Rewriting and Citation Validation](./section-rewriting-citation-validation.md) - Validation before commit
- [Workflow State Decoupling](./workflow-state-decoupling.md) - Recovery via store queries
- [Mandatory Archive Before Delete](../stores/mandatory-archive-before-delete.md) - Archive-first transaction semantics

## Known Uses in Thala

- `workflows/enhance/editing/document_model.py` - Transaction API and hierarchical anchors
- `workflows/enhance/editing/nodes/execute_edits.py` - Deep copy pattern for edit assembly
- `workflows/enhance/editing/nodes/plan_edits.py` - Edit deduplication
- `workflows/enhance/editing/nodes/finalize.py` - Final document rendering with header cleanup

## References

- [ACID Properties](https://en.wikipedia.org/wiki/ACID) - Database transaction semantics
- [Command Pattern](https://refactoring.guru/design-patterns/command) - Operation logging
- [Memento Pattern](https://refactoring.guru/design-patterns/memento) - State snapshot and restore
