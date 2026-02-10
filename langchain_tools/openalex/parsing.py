"""Data transformation functions for OpenAlex."""

from .models import OpenAlexAuthor, OpenAlexWork


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""

    # Build word->position mapping
    words_with_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words_with_positions.append((pos, word))

    # Sort by position and join
    words_with_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words_with_positions)


def _parse_work(work: dict) -> OpenAlexWork:
    """Parse OpenAlex work response into our model."""
    # Get open access info
    oa_info = work.get("open_access", {})
    oa_url = oa_info.get("oa_url")  # Best OA URL for full text
    is_oa = oa_info.get("is_oa", False)
    oa_status = oa_info.get("oa_status")

    # Get DOI (always keep for citations)
    doi = work.get("doi")

    # Prefer oa_url for scraping, fallback to DOI, then OpenAlex ID
    url = oa_url or doi or work.get("id", "")

    # Collect all unique OA URLs from locations array (best first)
    oa_urls: list[str] = []
    seen_urls: set[str] = set()
    if oa_url:
        oa_urls.append(oa_url)
        seen_urls.add(oa_url)

    for loc in work.get("locations", []):
        if not loc.get("is_oa"):
            continue
        for url_key in ("pdf_url", "landing_page_url"):
            loc_url = loc.get(url_key)
            if loc_url and loc_url not in seen_urls:
                oa_urls.append(loc_url)
                seen_urls.add(loc_url)

    # Extract PMC ID from ids object
    pmcid = None
    ids = work.get("ids", {})
    raw_pmcid = ids.get("pmcid")
    if raw_pmcid:
        # May be full URL like "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3429343"
        pmcid = raw_pmcid.split("/")[-1] if "/" in raw_pmcid else raw_pmcid

    # Parse authors (limit to first 5)
    authors = []
    for authorship in work.get("authorships", [])[:5]:
        author = authorship.get("author", {})
        institutions = authorship.get("institutions", [])
        institution_name = institutions[0].get("display_name") if institutions else None

        authors.append(
            OpenAlexAuthor(
                name=author.get("display_name", "Unknown"),
                institution=institution_name,
            )
        )

    # Get primary topic
    primary_topic = None
    topic_data = work.get("primary_topic")
    if topic_data:
        primary_topic = topic_data.get("display_name")

    # Get source/journal name
    source_name = None
    primary_location = work.get("primary_location", {})
    if primary_location:
        source = primary_location.get("source", {})
        if source:
            source_name = source.get("display_name")

    return OpenAlexWork(
        title=work.get("title") or work.get("display_name") or "Untitled",
        url=url,
        doi=doi,
        oa_url=oa_url,
        oa_urls=oa_urls,
        pmcid=pmcid,
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index", {})),
        authors=authors,
        publication_date=work.get("publication_date"),
        cited_by_count=work.get("cited_by_count", 0),
        primary_topic=primary_topic,
        source_name=source_name,
        is_oa=is_oa,
        oa_status=oa_status,
        language=work.get("language"),
    )
