---
name: substack-publishing-integration
title: "Substack Publishing Integration with Native Footnotes"
date: 2026-01-27
category: data-pipeline
applicability:
  - "Publishing illustrated markdown with citations to Substack"
  - "Converting citation formats to platform-native footnotes"
  - "Uploading local images to external platform storage"
components: [api_endpoint, configuration]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [substack, publishing, footnotes, prosemirror, markdown, citations, image-upload]
---

# Substack Publishing Integration with Native Footnotes

## Intent

Provide a complete pipeline for publishing illustrated markdown documents to Substack, converting academic-style `[@KEY]` citations to Substack's native hover-preview footnotes while handling image uploads, format cleanup, and multi-publication support.

## Motivation

Publishing research content to Substack presents several transformation challenges:

1. **Citation format mismatch**: Academic markdown uses `[@KEY]` citations with a References section, while Substack uses native footnotes with hover-preview
2. **Local images**: Markdown references local file paths that must be uploaded to Substack's S3
3. **Format artifacts**: YAML frontmatter, duplicate H1 titles, and malformed links need stripping
4. **ProseMirror schema**: Substack uses ProseMirror internally; the python-substack library has bugs that require post-processing
5. **Multi-publication support**: Users may publish to multiple Substack publications

This pattern addresses these challenges through a layered transformation pipeline with clear separation between conversion logic and publishing API.

## Applicability

Use this pattern when:
- You need to publish markdown with citations to Substack
- Local images must be uploaded to Substack's CDN
- Documents contain YAML frontmatter that should be stripped
- You want programmatic and CLI access to publishing
- You need to support multiple Substack publications

Do NOT use this pattern when:
- You're publishing to platforms other than Substack
- Documents don't contain citations or local images
- Manual copy-paste publishing is acceptable
- Real-time publishing (no review step) is required

## Structure

```
+-----------------------------------------------------------------------+
|                     Substack Publishing Pipeline                       |
+-----------------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------------+
|  Step 1: Preprocessing                                                |
|  --------------------                                                 |
|  - Strip YAML frontmatter (^---\n.*?\n---\n*)                        |
|  - Strip leading H1 titles (title is in post metadata)               |
|  - Parse References section to build key -> citation_text map         |
+-----------------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------------+
|  Step 2: Citation Processing                                          |
|  ---------------------------                                          |
|  - Extract citation order from body (first appearance = footnote 1)   |
|  - Build CitationMapping: {key, number, citation_text}                |
|  - Strip References section (content moves to footnotes)              |
+-----------------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------------+
|  Step 3: Image Upload                                                 |
|  -------------------                                                  |
|  - Find local images: ![alt](/path/to/image.jpg)                      |
|  - Upload each to Substack S3 via api.get_image()                     |
|  - Replace local paths with S3 URLs in markdown                       |
+-----------------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------------+
|  Step 4: ProseMirror Conversion                                       |
|  ------------------------------                                       |
|  - Convert markdown to ProseMirror via Post.from_markdown()           |
|  - Fix markdown links (href=null bug in python-substack)              |
|  - Convert --- paragraphs to horizontal_rule nodes                    |
|  - Inject footnoteAnchor nodes replacing [@KEY] in text               |
|  - Append footnote block nodes at document end                        |
+-----------------------------------------------------------------------+
                                |
                                v
+-----------------------------------------------------------------------+
|  Step 5: Publish/Draft                                                |
|  --------------------                                                 |
|  - Create Post object with title, subtitle, audience                  |
|  - POST to Substack API (draft or publish)                            |
|  - Return edit URL for review or publish URL                          |
+-----------------------------------------------------------------------+
```

## Implementation

### Step 1: Type Definitions

Define clear types for the pipeline stages:

```python
from typing import Literal, TypedDict


class SubstackConfig(TypedDict, total=False):
    """Configuration for Substack publishing."""

    cookies_path: str | None
    publication_url: str | None
    audience: Literal["everyone", "only_paid", "founding", "only_free"]


class CitationMapping(TypedDict):
    """Maps citation key to footnote number and content."""

    key: str  # e.g., "YFSXQJH4"
    number: int  # 1-indexed footnote number
    citation_text: str  # Full citation text for footnote


class ConversionResult(TypedDict):
    """Result of markdown to ProseMirror conversion."""

    draft_body: dict  # ProseMirror document structure
    title: str
    subtitle: str | None
    citations: list[CitationMapping]
    images_uploaded: list[str]  # S3 URLs
    warnings: list[str]


class PublishResult(TypedDict):
    """Result of publishing to Substack."""

    success: bool
    post_id: str | None
    draft_url: str | None
    publish_url: str | None
    error: str | None
```

### Step 2: Regex Patterns for Markdown Parsing

Define patterns for citation and content extraction:

```python
import re

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
```

### Step 3: Citation-to-Footnote Conversion

Extract citations and build footnote mappings:

```python
def parse_references_section(markdown: str) -> dict[str, str]:
    """Extract citation key to full citation text mapping from References section."""
    match = REFERENCES_HEADER_PATTERN.search(markdown)
    if not match:
        return {}

    references_text = markdown[match.end():]
    references = {}

    for entry_match in REFERENCE_ENTRY_PATTERN.finditer(references_text):
        key = entry_match.group(1)
        citation_text = entry_match.group(2).strip()
        citation_text = " ".join(citation_text.split())  # Normalize whitespace
        references[key] = citation_text

    return references


def extract_citation_order(markdown: str) -> list[str]:
    """Extract citation keys in order of first appearance (determines footnote numbering)."""
    # Strip References section first
    match = REFERENCES_HEADER_PATTERN.search(markdown)
    body = markdown[:match.start()] if match else markdown

    seen = set()
    ordered_keys = []

    for match in CITATION_PATTERN.finditer(body):
        citation_group = match.group(1)
        for key_match in KEY_PATTERN.finditer(citation_group):
            key = key_match.group(1)
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    return ordered_keys


def build_citation_mappings(
    citation_order: list[str],
    references: dict[str, str],
) -> list[CitationMapping]:
    """Build citation mappings with footnote numbers."""
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
```

### Step 4: ProseMirror Footnote Injection

Transform ProseMirror document to replace citations with footnotes:

```python
def inject_footnotes(
    draft_body: dict,
    citations: list[CitationMapping],
) -> dict:
    """Transform ProseMirror document to replace [@KEY] with footnotes.

    Substack footnote schema:
    - footnoteAnchor: inline node with {"number": int} attrs
    - footnote: block node with {"number": int} attrs and paragraph content
    """
    key_to_number = {c["key"]: c["number"] for c in citations}

    # Process document content
    new_content = []
    for node in draft_body.get("content", []):
        if node.get("type") == "paragraph":
            new_paragraph_content = _process_paragraph_for_citations(
                node.get("content", []),
                key_to_number,
            )
            new_content.append({
                "type": "paragraph",
                "content": new_paragraph_content,
            })
        else:
            new_content.append(node)

    # Append footnote definition blocks
    for citation in citations:
        new_content.append({
            "type": "footnote",
            "attrs": {"number": citation["number"]},
            "content": [{
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": citation["citation_text"]}
                ],
            }],
        })

    return {"type": "doc", "content": new_content}


def _process_paragraph_for_citations(
    content: list[dict],
    key_to_number: dict[str, int],
) -> list[dict]:
    """Process paragraph content, replacing citations with footnote anchors."""
    result = []

    for node in content:
        if node.get("type") != "text":
            result.append(node)
            continue

        text = node.get("text", "")
        marks = node.get("marks", [])

        last_end = 0
        for match in CITATION_PATTERN.finditer(text):
            # Add text before citation (strip trailing space before footnote)
            if match.start() > last_end:
                before_text = text[last_end:match.start()].rstrip(" ")
                if before_text:
                    text_node = {"type": "text", "text": before_text}
                    if marks:
                        text_node["marks"] = marks
                    result.append(text_node)

            # Parse citation keys and insert footnote anchors
            citation_group = match.group(1)
            keys = [m.group(1) for m in KEY_PATTERN.finditer(citation_group)]

            for key in keys:
                if key in key_to_number:
                    result.append({
                        "type": "footnoteAnchor",
                        "attrs": {"number": key_to_number[key]},
                    })

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
```

### Step 5: Publisher Class with Lazy-Loaded API

High-level interface for publishing:

```python
from substack import Api
from substack.post import Post


class SubstackPublisher:
    """High-level interface for publishing to Substack with footnotes and images."""

    def __init__(self, config: SubstackConfig):
        self.config = config
        self._api: Api | None = None

    @property
    def api(self) -> Api:
        """Lazy-load Substack API client."""
        if self._api is None:
            cookies_path = self.config.get("cookies_path")
            if cookies_path:
                self._api = Api(cookies_path=str(Path(cookies_path).expanduser()))
            else:
                raise ValueError("cookies_path required in config")

            # Set publication if specified
            publication_url = self.config.get("publication_url")
            if publication_url:
                pubs = self._api.get_user_publications()
                for pub in pubs:
                    if publication_url in pub.get("publication_url", ""):
                        self._api.change_publication(pub)
                        break

        return self._api

    def upload_image(self, local_path: str) -> str:
        """Upload a local image to Substack's S3."""
        result = self.api.get_image(local_path)
        return result.get("url", "") if isinstance(result, dict) else result

    def convert_markdown(
        self,
        markdown: str,
        title: str,
        subtitle: str | None = None,
        upload_images: bool = True,
    ) -> ConversionResult:
        """Convert markdown to Substack-ready ProseMirror document."""
        warnings = []

        # Step 1: Strip frontmatter and leading titles
        markdown = strip_frontmatter(markdown)
        markdown = strip_leading_titles(markdown)

        # Step 2: Parse references and extract citation order
        references = parse_references_section(markdown)
        citation_order = extract_citation_order(markdown)
        citations = build_citation_mappings(citation_order, references)

        # Step 3: Upload local images
        images_uploaded = []
        if upload_images:
            local_images = find_local_images(markdown)
            url_mapping = {}
            for _, local_path in local_images:
                s3_url = self.upload_image(local_path)
                url_mapping[local_path] = s3_url
                images_uploaded.append(s3_url)
            markdown = replace_image_urls(markdown, url_mapping)

        # Step 4: Strip references section and convert to ProseMirror
        markdown_stripped = strip_references_section(markdown)
        post = Post(title=title, subtitle=subtitle or "", user_id=self.user_id)
        post.from_markdown(markdown_stripped, self.api if upload_images else None)

        draft_body = post.draft_body

        # Step 5: Post-process ProseMirror document
        draft_body = convert_markdown_links(draft_body)  # Fix href=null bug
        if citations:
            draft_body = inject_footnotes(draft_body, citations)
        draft_body = convert_horizontal_rules(draft_body)

        return ConversionResult(
            draft_body=draft_body,
            title=title,
            subtitle=subtitle,
            citations=citations,
            images_uploaded=images_uploaded,
            warnings=warnings,
        )

    def create_draft(
        self,
        markdown: str,
        title: str,
        subtitle: str | None = None,
    ) -> PublishResult:
        """Create a draft post on Substack."""
        conversion = self.convert_markdown(markdown, title, subtitle)

        post = Post(
            title=title,
            subtitle=subtitle or "",
            user_id=self.user_id,
            audience=self.config.get("audience", "everyone"),
        )
        post.draft_body = conversion["draft_body"]

        result = self.api.post_draft(post.get_draft())

        post_id = result.get("id")
        pub_url = self.config.get("publication_url", "")
        draft_url = f"https://{pub_url}/publish/post/{post_id}" if post_id else None

        return PublishResult(
            success=True,
            post_id=post_id,
            draft_url=draft_url,
            publish_url=None,
            error=None,
        )
```

### Step 6: CLI Script

Provide command-line access:

```python
#!/usr/bin/env python3
"""Publish illustrated markdown to Substack with native footnotes."""

import argparse
from pathlib import Path

from utils.substack_publish import SubstackPublisher, SubstackConfig


def main():
    parser = argparse.ArgumentParser(
        description="Publish illustrated markdown to Substack with footnotes"
    )
    parser.add_argument("markdown_file", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--cookies", type=Path, default=None)
    parser.add_argument("--publication", default=None)
    parser.add_argument(
        "--audience",
        choices=["everyone", "only_paid", "founding", "only_free"],
        default="everyone",
    )

    args = parser.parse_args()

    config = SubstackConfig(
        cookies_path=str(args.cookies or "~/.substack-cookies.json"),
        publication_url=args.publication,
        audience=args.audience,
    )

    markdown = args.markdown_file.read_text(encoding="utf-8")
    publisher = SubstackPublisher(config)

    if args.draft_only:
        result = publisher.create_draft(markdown, args.title, args.subtitle)
        print(f"Draft URL: {result['draft_url']}")
    else:
        result = publisher.publish_post(markdown, args.title, args.subtitle)
        print(f"Published: {result['publish_url']}")


if __name__ == "__main__":
    main()
```

## Consequences

### Benefits

- **Native footnotes**: Citations render as Substack's hover-preview footnotes, not inline text
- **Image handling**: Local images are automatically uploaded and URLs replaced
- **Format cleanup**: YAML frontmatter and duplicate titles are stripped automatically
- **Multi-publication**: Supports publishing to different Substack publications
- **Dual interface**: Both programmatic API and CLI script available
- **Lazy loading**: API client only initialized when first needed

### Trade-offs

- **Cookie-based auth**: Requires browser cookie export (no OAuth support in python-substack)
- **ProseMirror complexity**: Must understand Substack's internal document format
- **Library bugs**: Requires workarounds for python-substack issues (href=null, etc.)
- **Citation format assumption**: Assumes specific `[@KEY]` format from Zotero workflow

### Alternatives

- **Manual publishing**: Copy-paste with manual footnote creation (doesn't scale)
- **Substack API directly**: More control but significantly more complex
- **Different footnote format**: Use numbered inline citations instead of native footnotes

## Related Patterns

- [Citation Processing with Zotero Integration](./citation-processing-zotero-integration.md) - Upstream pattern that generates the `[@KEY]` citations
- [Document Illustration Workflow](../langgraph/document-illustration-workflow.md) - Workflow that adds images before publishing

## Known Uses in Thala

- `utils/substack_publish/converter.py`: Markdown-to-ProseMirror conversion
- `utils/substack_publish/publisher.py`: High-level publishing interface
- `scripts/substack_publish.py`: CLI script for command-line publishing
- `utils/substack_publish/types.py`: TypedDict definitions for type safety

## References

- [python-substack library](https://github.com/ma2za/python-substack)
- [ProseMirror documentation](https://prosemirror.net/docs/)
- [Substack API (unofficial)](https://github.com/timothee-chauvin/substack-private-api)
