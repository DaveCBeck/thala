"""Researcher type definitions and allocation parsing."""

from typing_extensions import TypedDict


class ResearcherAllocation(TypedDict):
    """Allocation of researchers for a conduct_research action."""

    web_count: int  # Number of web researchers (1-3)


def parse_allocation(allocation_str: str) -> ResearcherAllocation:
    """Parse allocation string into ResearcherAllocation.

    Args:
        allocation_str: Single digit 1-3 for web researcher count

    Returns:
        ResearcherAllocation dict

    Raises:
        ValueError: If string is invalid format or out of range

    Examples:
        >>> parse_allocation("1")
        {'web_count': 1}
        >>> parse_allocation("3")
        {'web_count': 3}
    """
    if not allocation_str or len(allocation_str) != 1:
        raise ValueError(
            f"Allocation must be a single digit (1-3), got: {allocation_str!r}"
        )

    try:
        web = int(allocation_str)
    except ValueError:
        raise ValueError(f"Allocation must be a digit (1-3), got: {allocation_str!r}")

    if not 1 <= web <= 3:
        raise ValueError(f"Allocation must be 1-3, got: {allocation_str!r}")

    return ResearcherAllocation(web_count=web)
