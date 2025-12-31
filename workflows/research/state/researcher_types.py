"""Researcher type definitions and allocation parsing."""

from enum import Enum
from typing_extensions import TypedDict


class ResearcherType(str, Enum):
    """Types of specialized researchers."""

    WEB = "web"           # Firecrawl + Perplexity
    ACADEMIC = "academic" # OpenAlex
    BOOK = "book"         # book_search


class ResearcherAllocation(TypedDict):
    """Allocation of researchers for a conduct_research action."""

    web_count: int       # Number of web researchers (default: 1)
    academic_count: int  # Number of academic researchers (default: 1)
    book_count: int      # Number of book researchers (default: 1)


def parse_allocation(allocation_str: str) -> ResearcherAllocation:
    """Parse allocation string like '111', '210', '300' into ResearcherAllocation.

    Format: 3-character string where each digit represents count of:
    - Position 0: web researchers
    - Position 1: academic researchers
    - Position 2: book researchers

    Total must not exceed 3 (MAX_CONCURRENT_RESEARCHERS).

    Args:
        allocation_str: String like "111" (1 web, 1 academic, 1 book)
                       or "300" (3 web, 0 academic, 0 book)

    Returns:
        ResearcherAllocation dict

    Raises:
        ValueError: If string is invalid format or total exceeds 3

    Examples:
        >>> parse_allocation("111")
        {'web_count': 1, 'academic_count': 1, 'book_count': 1}
        >>> parse_allocation("210")
        {'web_count': 2, 'academic_count': 1, 'book_count': 0}
        >>> parse_allocation("300")
        {'web_count': 3, 'academic_count': 0, 'book_count': 0}
    """
    if not allocation_str or len(allocation_str) != 3:
        raise ValueError(
            f"Allocation must be a 3-character string (e.g., '111', '210'), got: {allocation_str!r}"
        )

    try:
        web = int(allocation_str[0])
        academic = int(allocation_str[1])
        book = int(allocation_str[2])
    except ValueError:
        raise ValueError(
            f"Allocation must contain only digits (0-3), got: {allocation_str!r}"
        )

    if not all(0 <= x <= 3 for x in [web, academic, book]):
        raise ValueError(
            f"Each allocation digit must be 0-3, got: {allocation_str!r}"
        )

    total = web + academic + book
    if total > 3:
        raise ValueError(
            f"Total allocation must not exceed 3, got {total} from '{allocation_str}'"
        )

    if total == 0:
        raise ValueError(
            f"Total allocation must be at least 1, got 0 from '{allocation_str}'"
        )

    return ResearcherAllocation(
        web_count=web,
        academic_count=academic,
        book_count=book,
    )
