"""URL download utilities."""

import logging

import httpx

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Base exception for download failures."""

    def __init__(self, message: str, url: str):
        self.url = url
        super().__init__(message)


class ContentTypeError(DownloadError):
    """Content did not match expected type."""

    pass


async def download_url(
    url: str,
    *,
    timeout: float = 60.0,
    expected_content_type: str | None = None,
    validate_pdf: bool = False,
) -> bytes:
    """Download content from URL.

    Args:
        url: URL to download
        timeout: Request timeout in seconds
        expected_content_type: If set, validates content-type header contains this string
            (e.g., "pdf", "html")
        validate_pdf: If True, validates content starts with PDF magic bytes (%PDF)

    Returns:
        Downloaded content as bytes

    Raises:
        httpx.HTTPStatusError: On HTTP error responses
        ContentTypeError: If content validation fails
        DownloadError: On other download failures
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

        content = response.content
        content_type = response.headers.get("content-type", "").lower()

        # Validate content-type if specified
        if expected_content_type and expected_content_type.lower() not in content_type:
            # For PDFs, also check magic bytes as fallback
            if expected_content_type.lower() == "pdf" and content[:4] == b"%PDF":
                pass  # Valid PDF despite content-type
            else:
                raise ContentTypeError(
                    f"Expected {expected_content_type}, got {content_type}",
                    url=url,
                )

        # Validate PDF magic bytes if requested
        if validate_pdf and content[:4] != b"%PDF":
            raise ContentTypeError(
                "Content does not start with PDF magic bytes",
                url=url,
            )

        return content
