"""Integration tests for OA PDF downloads via get_url().

These tests hit real publisher URLs to verify we can download genuine
Open Access PDFs. They require network access and a running Marker service.

Run with: pytest tests/integration/scraping/test_oa_pdf_downloads.py -m integration -v
"""

import pytest

from core.scraping import get_url, GetUrlOptions

# Confirmed-working OA URLs from a real lit review run (2026-02-07).
# Each was verified accessible in a browser.
OA_TEST_CASES = [
    pytest.param(
        "https://link.springer.com/content/pdf/10.1007/s12603-023-1895-1.pdf",
        "10.1007/s12603-023-1895-1",
        "Springer PDF direct",
        id="springer-pdf",
    ),
    pytest.param(
        "https://www.mdpi.com/1422-0067/20/22/5815/pdf?version=1574249998",
        "10.3390/ijms20225815",
        "MDPI PDF with version query param",
        id="mdpi-pdf-1",
    ),
    pytest.param(
        "https://www.mdpi.com/1422-0067/19/6/1687/pdf?version=1528273206",
        "10.3390/ijms19061687",
        "MDPI PDF with version query param",
        id="mdpi-pdf-2",
    ),
    pytest.param(
        "https://academic.oup.com/ageing/article-pdf/46/2/245/10802446/afw212.pdf",
        "10.1093/ageing/afw212",
        "Oxford University Press article-pdf",
        id="oup-pdf",
    ),
    pytest.param(
        "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1046/j.1474-9728.2003.00045.x",
        "10.1046/j.1474-9728.2003.00045.x",
        "Wiley pdfdirect",
        id="wiley-pdf",
    ),
    pytest.param(
        "https://www.frontiersin.org/articles/10.3389/fmolb.2015.00022/pdf",
        "10.3389/fmolb.2015.00022",
        "Frontiers article PDF",
        id="frontiers-pdf",
    ),
]


@pytest.mark.integration
@pytest.mark.asyncio
class TestOAPdfDownloads:
    """Test that get_url() can retrieve genuine Open Access PDFs."""

    @pytest.mark.parametrize("url,doi,description", OA_TEST_CASES)
    async def test_get_url_returns_content(self, url: str, doi: str, description: str):
        """get_url() should return markdown content for a genuine OA URL.

        Accepts content from any source (pdf_direct, scraped, etc.) —
        the point is that we get usable content, not how we got it.
        """
        result = await get_url(
            url,
            GetUrlOptions(
                detect_academic=True,
                allow_retrieve_academic=False,
            ),
        )

        assert result is not None, f"get_url returned None for {description} ({url})"
        assert len(result.content) > 500, (
            f"Content too short ({len(result.content)} chars) for {description}. "
            f"Source: {result.source}, provider: {result.provider}, "
            f"chain: {result.fallback_chain}"
        )
        # Should be markdown (not raw HTML or error page)
        assert "<html" not in result.content[:200].lower(), (
            f"Got raw HTML instead of markdown for {description}. "
            f"First 200 chars: {result.content[:200]}"
        )
