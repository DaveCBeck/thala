"""Substack publishing utility with native footnote support.

This module provides tools to publish illustrated markdown documents to Substack,
converting [@KEY] citations to native hover-preview footnotes.

Example usage:
    from utils.substack_publish import SubstackPublisher, SubstackConfig

    config = SubstackConfig(
        cookies_path=".thala/.substack-cookies.json",
        publication_url="mysubstack.substack.com",
    )

    publisher = SubstackPublisher(config)
    result = publisher.create_draft(
        markdown=article_content,
        title="My Article Title",
    )
    print(f"Draft created: {result['draft_url']}")
"""

from .converter import (
    build_citation_mappings,
    convert_horizontal_rules,
    convert_paywall_markers,
    extract_citation_order,
    find_local_images,
    inject_footnotes,
    parse_references_section,
    replace_image_urls,
    strip_frontmatter,
    strip_references_section,
)
from .publisher import SubstackPublisher
from .types import (
    CitationMapping,
    ConversionResult,
    PublishResult,
    SubstackConfig,
)

__all__ = [
    # Publisher
    "SubstackPublisher",
    # Types
    "SubstackConfig",
    "CitationMapping",
    "ConversionResult",
    "PublishResult",
    # Converter functions (for testing/advanced use)
    "parse_references_section",
    "extract_citation_order",
    "strip_references_section",
    "strip_frontmatter",
    "convert_horizontal_rules",
    "convert_paywall_markers",
    "find_local_images",
    "replace_image_urls",
    "build_citation_mappings",
    "inject_footnotes",
]
