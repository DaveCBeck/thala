"""Convert markdown with citations to Substack ProseMirror format with footnotes.

This module handles the transformation from illustrated markdown (with [@KEY] citations
and a ## References section) to Substack's ProseMirror document format with native
footnotes.

Substack footnote schema:
- footnoteAnchor: inline node with {"number": int} attrs
- footnote: block node with {"number": int} attrs and paragraph content
"""

import logging
import re
from pathlib import Path

from .types import CitationMapping

logger = logging.getLogger(__name__)

# Pattern to match [@KEY] or [@KEY1; @KEY2] citations
CITATION_PATTERN = re.compile(r"\[@([A-Z0-9]+(?:;\s*@[A-Z0-9]+)*)\]")

# Pattern to match individual keys within a citation group
KEY_PATTERN = re.compile(r"@?([A-Z0-9]{8})")

# Pattern to find the References section
REFERENCES_HEADER_PATTERN = re.compile(r"\n+---\n+## References\n+", re.MULTILINE)

# Pattern to parse individual reference entries: [@KEY] citation text
REFERENCE_ENTRY_PATTERN = re.compile(
    r"\[@([A-Z0-9]+)\]\s*(.+?)(?=\n\[@[A-Z0-9]+\]|\n*$)",
    re.DOTALL,
)

# Pattern to match local image paths
LOCAL_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\((/[^)]+)\)")

# Pattern to match markdown links [text](url)
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")

# Pattern to match YAML frontmatter
FRONTMATTER_PATTERN = re.compile(r"^---\n.*?\n---\n*", re.DOTALL)

# Pattern to match leading H1 titles (to strip duplicates)
LEADING_H1_PATTERN = re.compile(r"^(#\s+[^\n]+\n+)+", re.MULTILINE)

# Pattern to match horizontal rules (standalone --- on a line)
HORIZONTAL_RULE_PATTERN = re.compile(r"^\n*---\n*$", re.MULTILINE)


def strip_frontmatter(markdown: str) -> str:
    """Remove YAML frontmatter from markdown.

    Args:
        markdown: Markdown document potentially starting with ---...---

    Returns:
        Markdown with frontmatter removed
    """
    return FRONTMATTER_PATTERN.sub("", markdown)


def strip_leading_titles(markdown: str) -> str:
    """Remove leading H1 title lines from markdown.

    Since the title is set separately in Substack post metadata,
    we don't want duplicate titles at the top of the content.

    Args:
        markdown: Markdown document (frontmatter already stripped)

    Returns:
        Markdown with leading # Title lines removed
    """
    return LEADING_H1_PATTERN.sub("", markdown.lstrip())


def parse_references_section(markdown: str) -> dict[str, str]:
    """Extract citation key to full citation text mapping from References section.

    Args:
        markdown: Full markdown document with ## References section

    Returns:
        Dictionary mapping citation keys to their full citation text
    """
    # Find the References section
    match = REFERENCES_HEADER_PATTERN.search(markdown)
    if not match:
        logger.warning("No References section found in markdown")
        return {}

    # Get everything after "## References"
    references_text = markdown[match.end() :]

    # Parse each reference entry
    references = {}
    for entry_match in REFERENCE_ENTRY_PATTERN.finditer(references_text):
        key = entry_match.group(1)
        citation_text = entry_match.group(2).strip()
        # Clean up any trailing whitespace or newlines
        citation_text = " ".join(citation_text.split())
        references[key] = citation_text

    logger.info(f"Parsed {len(references)} references from markdown")
    return references


def extract_citation_order(markdown: str) -> list[str]:
    """Extract citation keys in order of first appearance in the body.

    This determines footnote numbering - first citation becomes footnote 1, etc.

    Args:
        markdown: Full markdown document

    Returns:
        List of unique citation keys in order of first appearance
    """
    # First, strip the References section to avoid counting citations there
    body = strip_references_section(markdown)

    # Find all citations in order
    seen = set()
    ordered_keys = []

    for match in CITATION_PATTERN.finditer(body):
        citation_group = match.group(1)
        # Handle multi-citations like [@KEY1; @KEY2]
        for key_match in KEY_PATTERN.finditer(citation_group):
            key = key_match.group(1)
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    logger.info(f"Found {len(ordered_keys)} unique citations in document body")
    return ordered_keys


def strip_references_section(markdown: str) -> str:
    """Remove the References section from markdown.

    Args:
        markdown: Full markdown document

    Returns:
        Markdown with References section removed
    """
    match = REFERENCES_HEADER_PATTERN.search(markdown)
    if match:
        # Remove from the "---" before References to end
        return markdown[: match.start()].rstrip()
    return markdown


def find_local_images(markdown: str) -> list[tuple[str, str]]:
    """Find all local image references in markdown.

    Args:
        markdown: Markdown document

    Returns:
        List of (alt_text, local_path) tuples
    """
    images = []
    for match in LOCAL_IMAGE_PATTERN.finditer(markdown):
        alt_text = match.group(1)
        local_path = match.group(2)
        if Path(local_path).exists():
            images.append((alt_text, local_path))
        else:
            logger.warning(f"Local image not found: {local_path}")
    return images


def replace_image_urls(markdown: str, url_mapping: dict[str, str]) -> str:
    """Replace local image paths with S3 URLs.

    Args:
        markdown: Markdown document
        url_mapping: Dict mapping local paths to S3 URLs

    Returns:
        Markdown with local paths replaced by S3 URLs
    """

    def replace_path(match: re.Match) -> str:
        alt_text = match.group(1)
        local_path = match.group(2)
        if local_path in url_mapping:
            return f"![{alt_text}]({url_mapping[local_path]})"
        return match.group(0)

    return LOCAL_IMAGE_PATTERN.sub(replace_path, markdown)


def build_citation_mappings(
    citation_order: list[str],
    references: dict[str, str],
) -> list[CitationMapping]:
    """Build citation mappings with footnote numbers.

    Args:
        citation_order: Keys in order of first appearance
        references: Key to citation text mapping

    Returns:
        List of CitationMapping with assigned footnote numbers
    """
    mappings = []
    for i, key in enumerate(citation_order, start=1):
        citation_text = references.get(key, f"[Reference not found: {key}]")
        mappings.append(
            CitationMapping(
                key=key,
                number=i,
                citation_text=citation_text,
            )
        )
    return mappings


def convert_horizontal_rules(draft_body: dict) -> dict:
    """Convert paragraphs containing only '---' to horizontal_rule nodes.

    Args:
        draft_body: ProseMirror document

    Returns:
        Document with horizontal rules converted
    """
    new_content = []
    for node in draft_body.get("content", []):
        if node.get("type") == "paragraph":
            content = node.get("content", [])
            # Check if paragraph is just "---"
            if (
                len(content) == 1
                and content[0].get("type") == "text"
                and content[0].get("text", "").strip() == "---"
            ):
                new_content.append({"type": "horizontal_rule"})
            else:
                new_content.append(node)
        else:
            new_content.append(node)
    return {"type": "doc", "content": new_content}


def inject_footnotes(
    draft_body: dict,
    citations: list[CitationMapping],
) -> dict:
    """Transform ProseMirror document to replace [@KEY] with footnotes.

    This function:
    1. Walks through all paragraph content
    2. Finds text nodes containing [@KEY] patterns
    3. Splits text nodes and inserts footnoteAnchor nodes
    4. Appends footnote definition blocks at document end

    Args:
        draft_body: ProseMirror document from Post.from_markdown()
        citations: List of citation mappings with footnote numbers

    Returns:
        Modified ProseMirror document with footnotes
    """
    # Build key to number mapping for quick lookup
    key_to_number = {c["key"]: c["number"] for c in citations}

    # Process document content
    new_content = []
    for node in draft_body.get("content", []):
        if node.get("type") == "paragraph":
            new_paragraph_content = _process_paragraph_for_citations(
                node.get("content", []),
                key_to_number,
            )
            new_content.append(
                {
                    "type": "paragraph",
                    "content": new_paragraph_content,
                }
            )
        else:
            # Pass through other node types unchanged
            new_content.append(node)

    # Append footnote definition blocks
    for citation in citations:
        new_content.append(
            {
                "type": "footnote",
                "attrs": {"number": citation["number"]},
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": citation["citation_text"]}
                        ],
                    }
                ],
            }
        )

    return {"type": "doc", "content": new_content}


def _process_paragraph_for_citations(
    content: list[dict],
    key_to_number: dict[str, int],
) -> list[dict]:
    """Process paragraph content, replacing citations with footnote anchors.

    Args:
        content: List of inline nodes (text, marks, etc.)
        key_to_number: Mapping of citation key to footnote number

    Returns:
        Modified content list with footnoteAnchor nodes inserted
    """
    result = []

    for node in content:
        if node.get("type") != "text":
            # Pass through non-text nodes (links, etc.)
            result.append(node)
            continue

        text = node.get("text", "")
        marks = node.get("marks", [])

        # Process text, splitting at citations
        last_end = 0
        for match in CITATION_PATTERN.finditer(text):
            # Add text before this citation (strip trailing space before footnote)
            if match.start() > last_end:
                before_text = text[last_end : match.start()].rstrip(" ")
                if before_text:
                    text_node = {"type": "text", "text": before_text}
                    if marks:
                        text_node["marks"] = marks
                    result.append(text_node)

            # Parse citation keys (may be multiple: [@K1; @K2])
            citation_group = match.group(1)
            keys = [m.group(1) for m in KEY_PATTERN.finditer(citation_group)]

            # Insert footnote anchor for each key
            for key in keys:
                if key in key_to_number:
                    result.append(
                        {
                            "type": "footnoteAnchor",
                            "attrs": {"number": key_to_number[key]},
                        }
                    )
                else:
                    # Key not in mappings - leave as text
                    logger.warning(f"Citation key not in mappings: {key}")
                    result.append({"type": "text", "text": f"[@{key}]"})

            last_end = match.end()

        # Add remaining text after last citation
        if last_end < len(text):
            after_text = text[last_end:]
            if after_text:
                text_node = {"type": "text", "text": after_text}
                if marks:
                    text_node["marks"] = marks
                result.append(text_node)

    return result


def convert_markdown_links(draft_body: dict) -> dict:
    """Convert markdown link syntax in text nodes to proper link marks.

    The python-substack from_markdown() has a bug where link hrefs are null.
    This function finds [text](url) patterns and converts them properly.

    Args:
        draft_body: ProseMirror document

    Returns:
        Document with markdown links converted to proper link nodes
    """
    new_content = []
    for node in draft_body.get("content", []):
        if node.get("type") == "paragraph":
            new_paragraph_content = _process_text_for_links(node.get("content", []))
            new_content.append({"type": "paragraph", "content": new_paragraph_content})
        else:
            new_content.append(node)
    return {"type": "doc", "content": new_content}


def _process_text_for_links(content: list[dict]) -> list[dict]:
    """Process paragraph content, converting [text](url) to link marks.

    Args:
        content: List of inline nodes

    Returns:
        Modified content with proper link nodes
    """
    result = []

    for node in content:
        if node.get("type") != "text":
            result.append(node)
            continue

        text = node.get("text", "")
        marks = node.get("marks", [])

        # Check if this text contains markdown links
        if not LINK_PATTERN.search(text):
            result.append(node)
            continue

        # Process text, splitting at links
        last_end = 0
        for match in LINK_PATTERN.finditer(text):
            # Add text before this link
            if match.start() > last_end:
                before_text = text[last_end : match.start()]
                if before_text:
                    text_node = {"type": "text", "text": before_text}
                    if marks:
                        text_node["marks"] = marks
                    result.append(text_node)

            # Create link node
            link_text = match.group(1)
            link_url = match.group(2)
            link_marks = marks.copy() if marks else []
            link_marks.append({"type": "link", "attrs": {"href": link_url}})
            result.append({"type": "text", "text": link_text, "marks": link_marks})

            last_end = match.end()

        # Add remaining text after last link
        if last_end < len(text):
            after_text = text[last_end:]
            if after_text:
                text_node = {"type": "text", "text": after_text}
                if marks:
                    text_node["marks"] = marks
                result.append(text_node)

    return result
