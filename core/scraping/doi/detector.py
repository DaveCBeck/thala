"""DOI detection from strings and URLs.

Handles multiple input formats:
1. Bare DOI: "10.1234/example"
2. DOI URL: "https://doi.org/10.1234/example"
3. Publisher URLs with embedded DOI
4. DOI in URL path or query params
5. DOI in page content
"""

import re
from typing import Callable, Optional
from urllib.parse import urlparse, unquote

from ..types import DoiInfo

# DOI regex pattern - matches the DOI itself
# DOIs start with 10. followed by registrant code (4+ digits) and suffix
DOI_REGEX = re.compile(r"10\.\d{4,}/[^\s<>\"'\]\),;]+")

# Publisher URL patterns that embed DOIs
# Each pattern maps to a function that extracts the DOI from the match
PUBLISHER_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Optional[str]]]] = [
    # Springer: link.springer.com/article/10.1007/s00234-023-01234-5
    (
        re.compile(r"link\.springer\.com/article/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # Wiley: onlinelibrary.wiley.com/doi/10.1002/abc.12345
    (
        re.compile(r"onlinelibrary\.wiley\.com/doi/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # PNAS: pnas.org/doi/10.1073/pnas.1234567890
    (
        re.compile(r"pnas\.org/doi/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # Oxford Academic: academic.oup.com/*/article/*/*/*
    (
        re.compile(r"academic\.oup\.com.*/doi/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # Science Direct (Elsevier): sciencedirect.com/science/article/pii/*
    # Note: PII is not DOI, need metadata lookup - return None
    (
        re.compile(r"sciencedirect\.com/science/article/pii/", re.IGNORECASE),
        lambda m: None,
    ),
    # Nature: nature.com/articles/s41586-020-2649-2
    # Nature article IDs are not DOIs - return None
    (
        re.compile(r"nature\.com/articles/[^/]+", re.IGNORECASE),
        lambda m: None,
    ),
    # Taylor & Francis: tandfonline.com/doi/full/10.1080/12345678
    (
        re.compile(r"tandfonline\.com/doi/(?:full|abs)/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # SAGE: journals.sagepub.com/doi/10.1177/12345678901234
    (
        re.compile(r"journals\.sagepub\.com/doi/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # ACS: pubs.acs.org/doi/10.1021/acs.joc.1c01234
    (
        re.compile(r"pubs\.acs\.org/doi/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # RSC: pubs.rsc.org/en/content/articlelanding/*/doi/10.1039/*
    (
        re.compile(r"pubs\.rsc\.org.*/doi/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # IOP Science: iopscience.iop.org/article/10.1088/1234-5678/12/3/456
    (
        re.compile(r"iopscience\.iop\.org/article/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
    # Cambridge: cambridge.org/core/journals/*/article/*/DOI
    (
        re.compile(r"cambridge\.org/core/.*/doi/(10\.\d+/[^/?]+)", re.IGNORECASE),
        lambda m: m.group(1),
    ),
]


def _normalize_doi(doi: str) -> str:
    """Clean up DOI by removing trailing punctuation."""
    # Remove trailing periods, commas, semicolons
    return doi.rstrip(".,;:")


def detect_doi(input_str: str) -> Optional[DoiInfo]:
    """Detect DOI from input string (bare DOI or URL).

    Handles:
    1. Bare DOI: "10.1234/example"
    2. DOI URL: "https://doi.org/10.1234/example"
    3. Publisher URLs with embedded DOI
    4. DOI in URL path or query params

    Args:
        input_str: URL, DOI, or doi.org URL

    Returns:
        DoiInfo if DOI detected, None otherwise
    """
    input_str = input_str.strip()

    # Check if input is a bare DOI (starts with 10.)
    if input_str.startswith("10."):
        match = DOI_REGEX.match(input_str)
        if match:
            doi = _normalize_doi(match.group(0))
            return DoiInfo(
                doi=doi,
                doi_url=f"https://doi.org/{doi}",
                source="input",
            )

    # Check if input is a URL
    try:
        parsed = urlparse(input_str)
        if not parsed.scheme:
            return None

        # Check doi.org URLs
        if "doi.org" in parsed.netloc:
            path = unquote(parsed.path.lstrip("/"))
            match = DOI_REGEX.search(path)
            if match:
                doi = _normalize_doi(match.group(0))
                return DoiInfo(
                    doi=doi,
                    doi_url=f"https://doi.org/{doi}",
                    source="url",
                )

        # Check publisher-specific patterns
        for pattern, extractor in PUBLISHER_PATTERNS:
            match = pattern.search(input_str)
            if match:
                doi = extractor(match)
                if doi:
                    doi = _normalize_doi(doi)
                    return DoiInfo(
                        doi=doi,
                        doi_url=f"https://doi.org/{doi}",
                        source="url",
                    )

        # Try to find DOI anywhere in URL
        match = DOI_REGEX.search(input_str)
        if match:
            doi = _normalize_doi(match.group(0))
            return DoiInfo(
                doi=doi,
                doi_url=f"https://doi.org/{doi}",
                source="url",
            )

    except Exception:
        pass

    return None


def extract_doi_from_content(markdown: str) -> Optional[str]:
    """Extract DOI from page content (metadata, citations, etc.).

    Looks for common DOI presentation patterns in scraped content.

    Args:
        markdown: Scraped markdown content

    Returns:
        DOI string if found, None otherwise
    """
    # Look for common DOI presentation patterns
    # Character class excludes: whitespace, brackets, quotes, parens
    patterns = [
        # Explicit DOI labels
        r"DOI:\s*(10\.\d{4,}/[^\s<>\"'\]\)]+)",
        r"doi:\s*(10\.\d{4,}/[^\s<>\"'\]\)]+)",
        # DOI URLs
        r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s<>\"'\]\)]+)",
        # Markdown links with DOI
        r"\[.*?\]\(https?://(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s<>\"'\]\)]+)\)",
    ]

    for pattern in patterns:
        match = re.search(pattern, markdown, re.IGNORECASE)
        if match:
            return _normalize_doi(match.group(1))

    return None
