#!/usr/bin/env python3
"""
Publish illustrated markdown to Substack with native footnotes.

This script converts markdown documents (with [@KEY] citations and a References
section) to Substack posts with hover-preview footnotes.

Usage:
    python scripts/substack_publish.py article.md --title "My Post"
    python scripts/substack_publish.py article.md --title "My Post" --draft-only

The script expects:
- Markdown with [@KEY] style citations in the body
- A "## References" section at the end with citation definitions
- Local image paths (which will be uploaded to Substack's S3)

Authentication (in priority order):
1. Email/password via --email/--password or SUBSTACK_EMAIL/SUBSTACK_PASSWORD env vars
2. Cookies via --cookies or SUBSTACK_COOKIES_PATH env var

Environment variables:
- SUBSTACK_EMAIL: Account email (recommended for multi-publication)
- SUBSTACK_PASSWORD: Account password
- SUBSTACK_COOKIES_PATH: Path to cookies JSON file (fallback)
- SUBSTACK_PUBLICATION_URL: Default publication URL (e.g., "mysubstack.substack.com")
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from core.task_queue.paths import SUBSTACK_COOKIES_FILE  # noqa: E402
from utils.substack_publish import SubstackConfig, SubstackPublisher  # noqa: E402

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
        "--email",
        default=None,
        help="Substack account email (or set SUBSTACK_EMAIL)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Substack account password (or set SUBSTACK_PASSWORD)",
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

    # Resolve cookies path (optional, used as fallback)
    cookies_path = args.cookies
    if cookies_path is None:
        env_cookies = os.environ.get("SUBSTACK_COOKIES_PATH")
        if env_cookies:
            cookies_path = Path(env_cookies).expanduser()
        elif SUBSTACK_COOKIES_FILE.exists():
            cookies_path = SUBSTACK_COOKIES_FILE

    # Resolve publication URL
    publication_url = args.publication
    if publication_url is None:
        publication_url = os.environ.get("SUBSTACK_PUBLICATION_URL")
    if publication_url is None:
        logger.error(
            "Publication URL required. Provide via --publication or SUBSTACK_PUBLICATION_URL"
        )
        sys.exit(1)

    # Check auth availability (warn if neither configured)
    email = args.email or os.environ.get("SUBSTACK_EMAIL")
    password = args.password or os.environ.get("SUBSTACK_PASSWORD")
    has_email_auth = email and password
    has_cookies = cookies_path and cookies_path.exists()

    if not has_email_auth and not has_cookies:
        logger.warning(
            "No authentication configured. "
            "Set SUBSTACK_EMAIL/SUBSTACK_PASSWORD or provide valid cookies."
        )

    # Build config (publisher handles auth cascade)
    config = SubstackConfig(
        email=args.email,
        password=args.password,
        cookies_path=str(cookies_path) if cookies_path and cookies_path.exists() else None,
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
