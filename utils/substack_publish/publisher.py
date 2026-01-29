"""Substack publishing wrapper using python-substack library.

This module provides a high-level interface for publishing markdown documents
to Substack, handling image uploads and footnote conversion.
"""

import logging
import os
from pathlib import Path

from substack import Api
from substack.post import Post

from .converter import (
    build_citation_mappings,
    convert_horizontal_rules,
    convert_markdown_links,
    convert_paywall_markers,
    extract_citation_order,
    find_local_images,
    inject_footnotes,
    parse_references_section,
    replace_image_urls,
    strip_frontmatter,
    strip_leading_titles,
    strip_references_section,
)
from .types import ConversionResult, PublishResult, SubstackConfig

logger = logging.getLogger(__name__)


class SubstackPublisher:
    """High-level interface for publishing to Substack with footnotes and images."""

    def __init__(self, config: SubstackConfig):
        """Initialize publisher with configuration.

        Args:
            config: SubstackConfig with email/password or cookies_path, publication_url, etc.
        """
        self.config = config
        self._api: Api | None = None

    def _create_api(self) -> Api:
        """Create API instance with authentication cascade.

        Priority:
        1. Email/password from config
        2. Email/password from environment
        3. Cookies from config
        4. Cookies from environment

        Returns:
            Authenticated Api instance

        Raises:
            ValueError: If no valid authentication method available
        """
        # Try email/password first (works for multi-publication)
        email = self.config.get("email") or os.getenv("SUBSTACK_EMAIL")
        password = self.config.get("password") or os.getenv("SUBSTACK_PASSWORD")

        if email and password:
            try:
                logger.info(f"Authenticating with email: {email}")
                api = Api(email=email, password=password)
                self._set_publication(api)
                return api
            except Exception as e:
                logger.warning(
                    f"Email/password auth failed: {e}. "
                    "Falling back to cookie authentication."
                )

        # Fall back to cookies
        cookies_path = self.config.get("cookies_path") or os.getenv("SUBSTACK_COOKIES_PATH")
        if cookies_path:
            path = Path(cookies_path).expanduser()
            if path.exists():
                logger.info(f"Authenticating with cookies: {path}")
                api = Api(cookies_path=str(path))
                self._set_publication(api)
                return api
            else:
                logger.warning(f"Cookies file not found: {path}")

        raise ValueError(
            "No valid Substack authentication. "
            "Set SUBSTACK_EMAIL/SUBSTACK_PASSWORD or provide cookies_path."
        )

    def _set_publication(self, api: Api) -> None:
        """Set the target publication on the API instance."""
        publication_url = self.config.get("publication_url") or os.getenv(
            "SUBSTACK_PUBLICATION_URL"
        )
        if not publication_url:
            return

        pubs = api.get_user_publications()
        subdomain = publication_url.replace(".substack.com", "")
        for pub in pubs:
            if (
                pub.get("subdomain") == subdomain
                or publication_url in pub.get("publication_url", "")
            ):
                api.change_publication(pub)
                logger.info(f"Set publication: {pub.get('name', subdomain)}")
                return

        logger.warning(f"Publication not found: {publication_url}")

    @property
    def api(self) -> Api:
        """Get or create API instance."""
        if self._api is None:
            self._api = self._create_api()
        return self._api

    @property
    def user_id(self) -> int:
        """Get the authenticated user's ID."""
        return self.api.get_user_id()

    def upload_image(self, local_path: str) -> str:
        """Upload a local image to Substack's S3.

        Args:
            local_path: Path to local image file

        Returns:
            S3 URL of uploaded image
        """
        logger.info(f"Uploading image: {local_path}")
        result = self.api.get_image(local_path)
        # get_image returns a dict with 'url' key
        if isinstance(result, dict):
            return result.get("url", "")
        return result

    def convert_markdown(
        self,
        markdown: str,
        title: str,
        subtitle: str | None = None,
        upload_images: bool = True,
    ) -> ConversionResult:
        """Convert markdown to Substack-ready ProseMirror document.

        This handles:
        1. Parsing the References section
        2. Uploading local images (if enabled)
        3. Converting markdown to ProseMirror via python-substack
        4. Injecting footnotes
        5. Stripping the References section

        Args:
            markdown: Full markdown document with citations and references
            title: Post title
            subtitle: Optional post subtitle
            upload_images: Whether to upload local images to S3

        Returns:
            ConversionResult with draft_body and metadata
        """
        warnings = []

        # Step 0: Strip YAML frontmatter if present
        markdown = strip_frontmatter(markdown)

        # Step 0b: Strip leading H1 titles (title is in post metadata)
        markdown = strip_leading_titles(markdown)

        # Step 1: Parse references from markdown
        references = parse_references_section(markdown)
        if not references:
            warnings.append("No references found in document")

        # Step 2: Extract citation order for footnote numbering
        citation_order = extract_citation_order(markdown)

        # Step 3: Build citation mappings
        citations = build_citation_mappings(citation_order, references)

        # Check for missing references
        for citation in citations:
            if citation["citation_text"].startswith("[Reference not found:"):
                warnings.append(f"Missing reference: {citation['key']}")

        # Step 4: Upload local images
        images_uploaded = []
        if upload_images:
            local_images = find_local_images(markdown)
            url_mapping = {}
            for _, local_path in local_images:
                try:
                    s3_url = self.upload_image(local_path)
                    url_mapping[local_path] = s3_url
                    images_uploaded.append(s3_url)
                    logger.info(f"Uploaded {local_path} -> {s3_url}")
                except Exception as e:
                    warnings.append(f"Failed to upload {local_path}: {e}")

            markdown = replace_image_urls(markdown, url_mapping)

        # Step 5: Strip references section (footnotes replace it)
        markdown_stripped = strip_references_section(markdown)

        # Step 6: Convert to ProseMirror using python-substack
        post = Post(title=title, subtitle=subtitle or "", user_id=self.user_id)

        # Use from_markdown with API for any remaining image handling
        post.from_markdown(markdown_stripped, self.api if upload_images else None)

        # Get the draft body
        draft_body = post.draft_body

        # Step 7: Fix markdown links (from_markdown has a bug with href=null)
        draft_body = convert_markdown_links(draft_body)

        # Step 8: Inject footnotes into the ProseMirror document
        if citations:
            draft_body = inject_footnotes(draft_body, citations)

        # Step 9: Convert horizontal rules (--- paragraphs to proper dividers)
        draft_body = convert_horizontal_rules(draft_body)

        # Step 10: Convert paywall markers (---paywall--- to paywall nodes)
        draft_body = convert_paywall_markers(draft_body)

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
        """Create a draft post on Substack.

        Args:
            markdown: Full markdown document
            title: Post title
            subtitle: Optional subtitle

        Returns:
            PublishResult with draft URL
        """
        try:
            # Convert markdown
            conversion = self.convert_markdown(markdown, title, subtitle)

            if conversion["warnings"]:
                for warning in conversion["warnings"]:
                    logger.warning(warning)

            # Create the post object
            audience = self.config.get("audience", "everyone")
            post = Post(
                title=title,
                subtitle=subtitle or "",
                user_id=self.user_id,
                audience=audience,
            )
            post.draft_body = conversion["draft_body"]

            # Create draft via API
            result = self.api.post_draft(post.get_draft())

            post_id = result.get("id")
            draft_url = None
            if post_id:
                # Construct draft URL
                pub_url = self.config.get("publication_url", "")
                if pub_url:
                    draft_url = f"https://{pub_url}/publish/post/{post_id}"

            return PublishResult(
                success=True,
                post_id=post_id,
                draft_url=draft_url,
                publish_url=None,
                error=None,
            )

        except Exception as e:
            logger.exception("Failed to create draft")
            return PublishResult(
                success=False,
                post_id=None,
                draft_url=None,
                publish_url=None,
                error=str(e),
            )

    def publish_post(
        self,
        markdown: str,
        title: str,
        subtitle: str | None = None,
    ) -> PublishResult:
        """Create and immediately publish a post on Substack.

        Args:
            markdown: Full markdown document
            title: Post title
            subtitle: Optional subtitle

        Returns:
            PublishResult with publish URL
        """
        try:
            # Convert markdown
            conversion = self.convert_markdown(markdown, title, subtitle)

            if conversion["warnings"]:
                for warning in conversion["warnings"]:
                    logger.warning(warning)

            # Create the post object
            audience = self.config.get("audience", "everyone")
            post = Post(
                title=title,
                subtitle=subtitle or "",
                user_id=self.user_id,
                audience=audience,
            )
            post.draft_body = conversion["draft_body"]

            # Publish via API
            result = self.api.publish_post(post.get_draft())

            post_id = result.get("id")
            publish_url = None
            if post_id:
                pub_url = self.config.get("publication_url", "")
                slug = result.get("slug", "")
                if pub_url and slug:
                    publish_url = f"https://{pub_url}/p/{slug}"

            return PublishResult(
                success=True,
                post_id=post_id,
                draft_url=None,
                publish_url=publish_url,
                error=None,
            )

        except Exception as e:
            logger.exception("Failed to publish post")
            return PublishResult(
                success=False,
                post_id=None,
                draft_url=None,
                publish_url=None,
                error=str(e),
            )
