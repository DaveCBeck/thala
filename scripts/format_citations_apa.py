#!/usr/bin/env python3
"""
Format Zotero citations in markdown documents to APA style.

This script:
1. Reads a markdown file with Zotero citation keys (e.g., [@AYDNQMFP])
2. Fetches item metadata from Zotero (via local-crud plugin)
3. Replaces inline citations with APA format (Author, Year)
4. Appends an APA-formatted bibliography at the end

Usage:
    python scripts/format_citations_apa.py input.md output.md
    python scripts/format_citations_apa.py input.md  # Outputs to input_apa.md
"""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stores.zotero import ZoteroStore, ZoteroItem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Regex patterns for citations
# Matches [@KEY] or [@KEY1; @KEY2; @KEY3]
CITATION_PATTERN = re.compile(r"\[@([A-Z0-9]+(?:;\s*@[A-Z0-9]+)*)\]")
# Matches individual keys within a multi-citation
KEY_PATTERN = re.compile(r"@?([A-Z0-9]{8})")


def extract_citation_keys(content: str) -> set[str]:
    """Extract all unique citation keys from markdown content."""
    keys = set()
    for match in CITATION_PATTERN.finditer(content):
        citation_group = match.group(1)
        for key_match in KEY_PATTERN.finditer(citation_group):
            keys.add(key_match.group(1))
    return keys


def format_author_name(creator: dict) -> str:
    """Format a single creator's name for APA style."""
    if creator.get("name"):
        # Single-field name (organization, etc.)
        return creator["name"]

    last_name = creator.get("lastName", "")
    first_name = creator.get("firstName", "")

    # Handle case where middle initials are incorrectly in lastName
    # e.g., firstName="Benjamin", lastName="A. Black" -> should be "Black, B. A."
    last_parts = last_name.split() if last_name else []
    leading_initials = []
    actual_last_name_parts = []

    for part in last_parts:
        stripped = part.rstrip(".")
        if len(stripped) <= 2 and stripped.isupper() and not actual_last_name_parts:
            # This is a leading initial incorrectly in the last name
            leading_initials.append(stripped + ".")
        else:
            actual_last_name_parts.append(part)

    actual_last_name = " ".join(actual_last_name_parts) if actual_last_name_parts else last_name

    if first_name:
        # APA: "LastName, F. M."
        initials = ". ".join(n[0].upper() for n in first_name.split() if n) + "."
        # Append any leading initials that were in the wrong field
        if leading_initials:
            initials = initials + " " + " ".join(leading_initials)
        return f"{actual_last_name}, {initials}"

    return actual_last_name


def get_clean_last_name(creator: dict) -> str:
    """Get the clean last name for a creator, handling misplaced initials."""
    last_name = creator.get("lastName", "") or creator.get("name", "")
    return normalize_last_name_for_sort(last_name) or last_name or "Unknown"


def format_authors_inline(creators: list[dict]) -> str:
    """Format author names for inline citation (Author, Year)."""
    authors = [c for c in creators if c.get("creatorType") in ("author", None)]

    if not authors:
        return "Unknown"

    if len(authors) == 1:
        return get_clean_last_name(authors[0])
    elif len(authors) == 2:
        name1 = get_clean_last_name(authors[0])
        name2 = get_clean_last_name(authors[1])
        return f"{name1} & {name2}"
    else:
        return f"{get_clean_last_name(authors[0])} et al."


def format_authors_bibliography(creators: list[dict]) -> str:
    """Format author names for bibliography entry."""
    authors = [c for c in creators if c.get("creatorType") in ("author", None)]

    if not authors:
        return "Unknown"

    formatted = []
    for i, author in enumerate(authors):
        name = format_author_name(author)
        formatted.append(name)

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    elif len(formatted) <= 20:
        # APA 7th: List all authors up to 20
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    else:
        # APA 7th: First 19, ellipsis, last author
        return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def extract_year(item: ZoteroItem) -> str:
    """Extract publication year from Zotero item."""
    date = item.fields.get("date", "")
    if not date:
        return "n.d."

    # Try to extract 4-digit year
    year_match = re.search(r"(\d{4})", date)
    if year_match:
        return year_match.group(1)

    return date if date else "n.d."


def format_inline_citation(item: ZoteroItem) -> str:
    """Format a single item as an inline APA citation."""
    author = format_authors_inline(item.creators)
    year = extract_year(item)
    return f"{author}, {year}"


def format_bibliography_entry(item: ZoteroItem) -> str:
    """Format a Zotero item as an APA bibliography entry."""
    authors = format_authors_bibliography(item.creators)
    year = extract_year(item)
    title = item.fields.get("title", "Untitled")

    item_type = item.itemType

    if item_type == "journalArticle":
        # Author, A. A. (Year). Title of article. Journal Name, Volume(Issue), pages. DOI
        journal = item.fields.get("publicationTitle", "")
        volume = item.fields.get("volume", "")
        issue = item.fields.get("issue", "")
        pages = item.fields.get("pages", "")
        doi = item.fields.get("DOI", "")

        entry = f"{authors} ({year}). {title}."
        if journal:
            entry += f" *{journal}*"
            if volume:
                entry += f", *{volume}*"
                if issue:
                    entry += f"({issue})"
            if pages:
                entry += f", {pages}"
            entry += "."
        if doi:
            entry += f" https://doi.org/{doi}"

        return entry

    elif item_type == "book":
        # Author, A. A. (Year). Title of work: Capital letter also for subtitle. Publisher.
        publisher = item.fields.get("publisher", "")

        entry = f"{authors} ({year}). *{title}*."
        if publisher:
            entry += f" {publisher}."

        return entry

    elif item_type == "bookSection":
        # Author, A. A. (Year). Title of chapter. In E. E. Editor (Ed.), Title of book (pp. x-x). Publisher.
        book_title = item.fields.get("bookTitle", "")
        publisher = item.fields.get("publisher", "")
        pages = item.fields.get("pages", "")

        # Get editors
        editors = [c for c in item.creators if c.get("creatorType") == "editor"]
        editor_str = ""
        if editors:
            editor_names = [format_author_name(e) for e in editors]
            if len(editor_names) == 1:
                editor_str = f"In {editor_names[0]} (Ed.), "
            else:
                editor_str = f"In {', '.join(editor_names[:-1])} & {editor_names[-1]} (Eds.), "

        entry = f"{authors} ({year}). {title}. {editor_str}*{book_title}*"
        if pages:
            entry += f" (pp. {pages})"
        entry += "."
        if publisher:
            entry += f" {publisher}."

        return entry

    elif item_type == "conferencePaper":
        # Author, A. A. (Year). Title of paper. Conference Name, Location.
        conference = item.fields.get("conferenceName", item.fields.get("proceedingsTitle", ""))

        entry = f"{authors} ({year}). {title}."
        if conference:
            entry += f" *{conference}*."

        return entry

    elif item_type == "thesis":
        # Author, A. A. (Year). Title of dissertation/thesis [Doctoral dissertation/Master's thesis, University]. Database.
        university = item.fields.get("university", "")
        thesis_type = item.fields.get("thesisType", "Thesis")

        entry = f"{authors} ({year}). *{title}* [{thesis_type}"
        if university:
            entry += f", {university}"
        entry += "]."

        return entry

    elif item_type == "webpage":
        # Author, A. A. (Year, Month Day). Title of page. Site Name. URL
        url = item.fields.get("url", "")
        site = item.fields.get("websiteTitle", "")

        entry = f"{authors} ({year}). {title}."
        if site:
            entry += f" *{site}*."
        if url:
            entry += f" {url}"

        return entry

    elif item_type == "report":
        # Author, A. A. (Year). Title of report (Report No. xxx). Publisher.
        report_number = item.fields.get("reportNumber", "")
        institution = item.fields.get("institution", "")

        entry = f"{authors} ({year}). *{title}*"
        if report_number:
            entry += f" (Report No. {report_number})"
        entry += "."
        if institution:
            entry += f" {institution}."

        return entry

    elif item_type == "preprint":
        # Author, A. A. (Year). Title of preprint. Repository. DOI/URL
        repository = item.fields.get("repository", "")
        doi = item.fields.get("DOI", "")
        url = item.fields.get("url", "")

        entry = f"{authors} ({year}). {title}."
        if repository:
            entry += f" *{repository}*."
        if doi:
            entry += f" https://doi.org/{doi}"
        elif url:
            entry += f" {url}"

        return entry

    else:
        # Generic format for other types
        entry = f"{authors} ({year}). {title}."

        # Add URL if available
        url = item.fields.get("url", "")
        doi = item.fields.get("DOI", "")
        if doi:
            entry += f" https://doi.org/{doi}"
        elif url:
            entry += f" {url}"

        return entry


def normalize_last_name_for_sort(last_name: str) -> str:
    """
    Normalize a last name for sorting purposes.

    Handles cases where middle initials are incorrectly stored in the lastName field.
    E.g., "A. Black" -> "Black", "N. Norris" -> "Norris", "von Strandmann" -> "von Strandmann"
    """
    if not last_name:
        return ""

    # Split on spaces and find the first non-initial part
    # An initial is typically 1-2 chars followed by a period, or just 1-2 uppercase letters
    parts = last_name.split()
    for part in parts:
        # Skip parts that look like initials (e.g., "A.", "A", "AB.")
        stripped = part.rstrip(".")
        if len(stripped) <= 2 and stripped.isupper():
            continue
        # This looks like an actual name part
        return " ".join(parts[parts.index(part):])

    # If all parts look like initials, return the original
    return last_name


def get_sort_key(item: ZoteroItem) -> tuple:
    """Get sort key for bibliography (author last name, year, title)."""
    authors = [c for c in item.creators if c.get("creatorType") in ("author", None)]
    first_author = ""
    if authors:
        last_name = authors[0].get("lastName", "") or authors[0].get("name", "")
        first_author = normalize_last_name_for_sort(last_name)

    year = extract_year(item)
    title = item.fields.get("title", "")

    return (first_author.lower(), year, title.lower())


async def fetch_zotero_items(
    store: ZoteroStore, keys: set[str]
) -> dict[str, Optional[ZoteroItem]]:
    """Fetch all Zotero items for the given keys."""
    items = {}
    total = len(keys)

    for i, key in enumerate(sorted(keys), 1):
        logger.info(f"Fetching item {i}/{total}: {key}")
        try:
            item = await store.get(key)
            items[key] = item
            if item is None:
                logger.warning(f"Item not found: {key}")
        except Exception as e:
            logger.error(f"Error fetching {key}: {e}")
            items[key] = None

    return items


def replace_citations(content: str, items: dict[str, Optional[ZoteroItem]]) -> str:
    """Replace citation keys with APA inline citations."""

    def replace_citation(match: re.Match) -> str:
        citation_group = match.group(1)
        keys = [m.group(1) for m in KEY_PATTERN.finditer(citation_group)]

        citations = []
        for key in keys:
            item = items.get(key)
            if item:
                citations.append(format_inline_citation(item))
            else:
                # Keep original key if item not found
                citations.append(f"@{key}")

        return "(" + "; ".join(citations) + ")"

    return CITATION_PATTERN.sub(replace_citation, content)


def generate_bibliography(items: dict[str, Optional[ZoteroItem]]) -> str:
    """Generate APA bibliography from fetched items."""
    # Filter to only items that were found
    found_items = [item for item in items.values() if item is not None]

    # Sort by author, year, title
    sorted_items = sorted(found_items, key=get_sort_key)

    # Format each entry
    entries = []
    for item in sorted_items:
        entry = format_bibliography_entry(item)
        entries.append(entry)

    # Build bibliography section
    bibliography = "\n\n---\n\n## References\n\n"
    for entry in entries:
        bibliography += f"{entry}\n\n"

    return bibliography


async def process_document(input_path: Path, output_path: Path) -> None:
    """Process a markdown document, replacing citations and adding bibliography."""

    # Read input file
    logger.info(f"Reading: {input_path}")
    content = input_path.read_text(encoding="utf-8")

    # Extract all citation keys
    keys = extract_citation_keys(content)
    logger.info(f"Found {len(keys)} unique citation keys")

    if not keys:
        logger.warning("No citations found in document")
        output_path.write_text(content, encoding="utf-8")
        return

    # Fetch items from Zotero
    async with ZoteroStore() as store:
        # Check connection
        health = await store.health_check()
        if not health.healthy:
            logger.error(f"Zotero not available: {health.error}")
            raise RuntimeError(f"Cannot connect to Zotero: {health.error}")

        logger.info(f"Connected to Zotero {health.zoteroVersion}")

        # Fetch all items
        items = await fetch_zotero_items(store, keys)

    # Count found vs missing
    found = sum(1 for v in items.values() if v is not None)
    missing = len(items) - found
    logger.info(f"Found {found}/{len(items)} items ({missing} missing)")

    # Replace citations in content
    processed_content = replace_citations(content, items)

    # Generate and append bibliography
    bibliography = generate_bibliography(items)
    processed_content += bibliography

    # Write output
    logger.info(f"Writing: {output_path}")
    output_path.write_text(processed_content, encoding="utf-8")

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Format Zotero citations in markdown to APA style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input markdown file with Zotero citations",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output file (default: input_apa.md)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Set default output path
    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + "_apa")

    # Run async processing
    asyncio.run(process_document(args.input, args.output))


if __name__ == "__main__":
    main()
