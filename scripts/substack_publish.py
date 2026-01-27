#!/usr/bin/env python3
"""
Publish illustrated markdown to Substack with native footnotes.

This script converts markdown documents (with [@KEY] citations and a References
section) to Substack posts with hover-preview footnotes.

Usage:
    python scripts/substack_publish.py article.md --title "My Post"
    python scripts/substack_publish.py article.md --title "My Post" --draft-only
    python scripts/substack_publish.py article.md --title "My Post" --cookies ~/.substack.json

The script expects:
- Markdown with [@KEY] style citations in the body
- A "## References" section at the end with citation definitions
- Local image paths (which will be uploaded to Substack's S3)

Environment variables:
- SUBSTACK_COOKIES_PATH: Default path to cookies JSON file
- SUBSTACK_PUBLICATION_URL: Default publication URL (e.g., "mysubstack.substack.com")
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.task_queue.paths import SUBSTACK_COOKIES_FILE
from utils.substack_publish import SubstackConfig, SubstackPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Publish illustrated markdown to Substack with footnotes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "markdown_file",
        type=Path,
        help="Path to markdown file to publish",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Post title",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Post subtitle (optional)",
    )
    parser.add_argument(
        "--draft-only",
        action="store_true",
        help="Create draft only, don't publish",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Path to Substack cookies JSON file",
    )
    parser.add_argument(
        "--publication",
        default=None,
        help="Publication URL (e.g., 'mysubstack.substack.com')",
    )
    parser.add_argument(
        "--audience",
        choices=["everyone", "only_paid", "founding", "only_free"],
        default="everyone",
        help="Post audience (default: everyone)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file
    if not args.markdown_file.exists():
        logger.error(f"Markdown file not found: {args.markdown_file}")
        sys.exit(1)

    # Resolve cookies path
    cookies_path = args.cookies
    if cookies_path is None:
        cookies_path = os.environ.get("SUBSTACK_COOKIES_PATH")
    if cookies_path is None:
        cookies_path = SUBSTACK_COOKIES_FILE
    else:
        cookies_path = Path(cookies_path).expanduser()

    if not cookies_path.exists():
        logger.error(
            f"Cookies file not found: {cookies_path}\n"
            "Export cookies from your browser while logged into Substack.\n"
            "See: https://github.com/ma2za/python-substack#authentication"
        )
        sys.exit(1)

    # Resolve publication URL
    publication_url = args.publication
    if publication_url is None:
        publication_url = os.environ.get("SUBSTACK_PUBLICATION_URL")
    if publication_url is None:
        logger.error(
            "Publication URL required. Provide via --publication or SUBSTACK_PUBLICATION_URL"
        )
        sys.exit(1)

    # Build config
    config = SubstackConfig(
        cookies_path=str(cookies_path),
        publication_url=publication_url,
        audience=args.audience,
    )

    # Read markdown
    logger.info(f"Reading: {args.markdown_file}")
    markdown = args.markdown_file.read_text(encoding="utf-8")

    # Create publisher and publish
    publisher = SubstackPublisher(config)

    if args.draft_only:
        logger.info("Creating draft...")
        result = publisher.create_draft(
            markdown=markdown,
            title=args.title,
            subtitle=args.subtitle,
        )
    else:
        logger.info("Publishing...")
        result = publisher.publish_post(
            markdown=markdown,
            title=args.title,
            subtitle=args.subtitle,
        )

    # Report result
    if result["success"]:
        if args.draft_only:
            logger.info(f"Draft created: {result['draft_url']}")
            print(f"\nDraft URL: {result['draft_url']}")
        else:
            logger.info(f"Published: {result['publish_url']}")
            print(f"\nPublished: {result['publish_url']}")
    else:
        logger.error(f"Failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
