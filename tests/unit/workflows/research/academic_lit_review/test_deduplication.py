"""Tests for paper deduplication logic including title+author matching."""

from workflows.research.academic_lit_review.utils.conversion import (
    _extract_author_lastnames,
    _is_title_author_duplicate,
    _normalize_title,
    _pick_preferred_version,
    deduplicate_papers,
)


def _make_paper(doi, title="Test Paper", authors=None, cited_by_count=0, year=2025):
    """Create a minimal PaperMetadata dict for testing."""
    return {
        "doi": doi,
        "title": title,
        "authors": authors or [],
        "publication_date": f"{year}-01-01",
        "year": year,
        "venue": None,
        "cited_by_count": cited_by_count,
        "abstract": None,
        "openalex_id": f"W{hash(doi) % 10**10}",
        "primary_topic": None,
        "is_oa": False,
        "oa_url": None,
        "oa_urls": [],
        "pmcid": None,
        "oa_status": None,
        "referenced_works": [],
        "citing_works_count": cited_by_count,
        "retrieved_at": None,
        "discovery_stage": 0,
        "discovery_method": "keyword",
        "relevance_score": None,
    }


def _authors(*names):
    """Create author dicts from full name strings."""
    return [{"name": n, "author_id": None, "institution": None, "orcid": None} for n in names]


class TestNormalizeTitle:
    def test_lowercases(self):
        assert _normalize_title("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalize_title("AI Agents vs. Agentic AI: A Conceptual Taxonomy") == \
               "ai agents vs agentic ai a conceptual taxonomy"

    def test_normalizes_accents(self):
        assert _normalize_title("Résumé of Naïve Approaches") == "resume of naive approaches"

    def test_collapses_whitespace(self):
        assert _normalize_title("Hello   World") == "hello world"

    def test_handles_empty(self):
        assert _normalize_title("") == ""


class TestExtractAuthorLastnames:
    def test_extracts_surnames(self):
        paper = _make_paper("10.1/a", authors=_authors("John Smith", "Jane Doe"))
        assert _extract_author_lastnames(paper) == ["smith", "doe"]

    def test_limits_to_n(self):
        paper = _make_paper("10.1/a", authors=_authors("A One", "B Two", "C Three", "D Four"))
        assert _extract_author_lastnames(paper, limit=2) == ["one", "two"]

    def test_handles_empty_authors(self):
        paper = _make_paper("10.1/a", authors=[])
        assert _extract_author_lastnames(paper) == []


class TestIsTitleAuthorDuplicate:
    def test_same_paper_different_case(self):
        """The Sapkota case: same title with different capitalization."""
        a = _make_paper(
            "10.1016/j.inffus.2025.01",
            title="AI Agents vs. Agentic AI: A Conceptual taxonomy, applications and challenges",
            authors=_authors("Ranjan Sapkota", "Konstantinos Roumeliotis", "Manoj Karkee"),
        )
        b = _make_paper(
            "10.3390/superintelligence.2025.01",
            title="AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges",
            authors=_authors("Ranjan Sapkota", "Konstantinos Roumeliotis", "Manoj Karkee"),
        )
        assert _is_title_author_duplicate(a, b) is True

    def test_different_papers_same_authors(self):
        a = _make_paper("10.1/a", title="Paper One", authors=_authors("John Smith"))
        b = _make_paper("10.1/b", title="Paper Two", authors=_authors("John Smith"))
        assert _is_title_author_duplicate(a, b) is False

    def test_same_title_different_authors(self):
        a = _make_paper("10.1/a", title="Survey of AI", authors=_authors("John Smith"))
        b = _make_paper("10.1/b", title="Survey of AI", authors=_authors("Jane Doe"))
        assert _is_title_author_duplicate(a, b) is False

    def test_no_authors_no_match(self):
        a = _make_paper("10.1/a", title="Same Title", authors=[])
        b = _make_paper("10.1/b", title="Same Title", authors=[])
        assert _is_title_author_duplicate(a, b) is False

    def test_single_shared_author_sufficient_when_solo(self):
        """When papers have only one author each, one match suffices."""
        a = _make_paper("10.1/a", title="My Paper", authors=_authors("Solo Author"))
        b = _make_paper("10.1/b", title="My Paper", authors=_authors("Solo Author"))
        assert _is_title_author_duplicate(a, b) is True


class TestPickPreferredVersion:
    def test_prefers_higher_citations(self):
        published = _make_paper("10.1/pub", cited_by_count=50)
        preprint = _make_paper("10.1/pre", cited_by_count=5)
        assert _pick_preferred_version(published, preprint) is published
        assert _pick_preferred_version(preprint, published) is published

    def test_tiebreaker_year(self):
        a = _make_paper("10.1/a", cited_by_count=10, year=2024)
        b = _make_paper("10.1/b", cited_by_count=10, year=2025)
        assert _pick_preferred_version(a, b) is b

    def test_final_tiebreaker_first_arg(self):
        a = _make_paper("10.1/a", cited_by_count=10, year=2025)
        b = _make_paper("10.1/b", cited_by_count=10, year=2025)
        assert _pick_preferred_version(a, b) is a


class TestDeduplicatePapers:
    def test_doi_dedup(self):
        papers = [
            _make_paper("10.1/a", title="Paper A"),
            _make_paper("10.1/a", title="Paper A (dup)"),
            _make_paper("10.1/b", title="Paper B"),
        ]
        result = deduplicate_papers(papers)
        assert len(result) == 2
        assert result[0]["doi"] == "10.1/a"
        assert result[1]["doi"] == "10.1/b"

    def test_doi_dedup_excludes_existing(self):
        papers = [_make_paper("10.1/a"), _make_paper("10.1/b")]
        result = deduplicate_papers(papers, existing_dois={"10.1/a"})
        assert len(result) == 1
        assert result[0]["doi"] == "10.1/b"

    def test_title_author_dedup_within_batch(self):
        """Preprint and published version in same batch — keep higher citations."""
        preprint = _make_paper(
            "10.48550/arXiv.2025.001",
            title="My Great Paper",
            authors=_authors("Alice Smith", "Bob Jones"),
            cited_by_count=2,
        )
        published = _make_paper(
            "10.1016/j.example.2025.001",
            title="My Great Paper",
            authors=_authors("Alice Smith", "Bob Jones"),
            cited_by_count=20,
        )
        result = deduplicate_papers([preprint, published])
        assert len(result) == 1
        assert result[0]["doi"] == "10.1016/j.example.2025.001"

    def test_title_author_dedup_against_existing_corpus(self):
        """Published version already in corpus, preprint discovered later."""
        existing = _make_paper(
            "10.1016/j.example.2025.001",
            title="My Great Paper",
            authors=_authors("Alice Smith", "Bob Jones"),
            cited_by_count=20,
        )
        preprint = _make_paper(
            "10.48550/arXiv.2025.001",
            title="My Great Paper",
            authors=_authors("Alice Smith", "Bob Jones"),
            cited_by_count=2,
        )
        result = deduplicate_papers(
            [preprint],
            existing_papers=[existing],
        )
        assert len(result) == 0

    def test_different_papers_not_deduplicated(self):
        a = _make_paper("10.1/a", title="Paper One", authors=_authors("Alice"))
        b = _make_paper("10.1/b", title="Paper Two", authors=_authors("Bob"))
        result = deduplicate_papers([a, b])
        assert len(result) == 2

    def test_backward_compatible_no_existing_papers(self):
        """Existing call sites that don't pass existing_papers still work."""
        papers = [_make_paper("10.1/a"), _make_paper("10.1/b")]
        result = deduplicate_papers(papers, existing_dois={"10.1/a"})
        assert len(result) == 1
