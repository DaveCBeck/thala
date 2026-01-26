"""
Tests for metadata validation and normalization utilities.
"""

from workflows.shared.metadata_utils import (
    extract_year,
    validate_year,
    parse_author_name,
    normalize_author_list,
    merge_metadata_with_baseline,
)


class TestYearExtraction:
    """Tests for extract_year function."""

    def test_valid_4_digit_year(self):
        assert extract_year("2023") == 2023
        assert extract_year("1999") == 1999
        assert extract_year("1500") == 1500

    def test_year_from_full_date(self):
        assert extract_year("2023-05-15") == 2023
        assert extract_year("1850-01-01") == 1850

    def test_year_from_text(self):
        assert extract_year("May 2023") == 2023
        assert extract_year("Published in 1876") == 1876
        assert extract_year("Winter 2024") == 2024

    def test_year_at_boundaries(self):
        # Valid range: 1500-2099
        assert extract_year("1500") == 1500
        assert extract_year("2099") == 2099

    def test_year_outside_range_returns_none(self):
        assert extract_year("1499") is None  # Too old
        assert extract_year("2100") is None  # Too future
        assert extract_year("1234") is None  # Before 1500

    def test_invalid_year_returns_none(self):
        assert extract_year("forthcoming") is None
        assert extract_year("in press") is None
        assert extract_year("") is None
        assert extract_year(None) is None
        assert extract_year("no date") is None

    def test_year_with_surrounding_text(self):
        assert extract_year("Copyright 2023 All Rights Reserved") == 2023
        assert extract_year("Vol. 15, 1987, pp. 123-145") == 1987


class TestYearValidation:
    """Tests for validate_year function."""

    def test_valid_year_returns_string(self):
        assert validate_year("2023") == "2023"
        assert validate_year("1850") == "1850"

    def test_invalid_year_returns_none(self):
        assert validate_year("forthcoming") is None
        assert validate_year("") is None
        assert validate_year(None) is None

    def test_extracts_from_date_string(self):
        assert validate_year("2023-05-15") == "2023"


class TestAuthorNameParsing:
    """Tests for parse_author_name function."""

    def test_simple_first_last(self):
        result = parse_author_name("John Smith")
        assert result.firstName == "John"
        assert result.lastName == "Smith"
        assert result.name is None

    def test_with_middle_name(self):
        result = parse_author_name("John Michael Smith")
        assert result.firstName == "John Michael"
        assert result.lastName == "Smith"

    def test_with_middle_initial(self):
        result = parse_author_name("John A. Smith")
        assert result.firstName == "John A."
        assert result.lastName == "Smith"

    def test_multiple_initials(self):
        result = parse_author_name("J. R. R. Tolkien")
        assert result.firstName == "J. R. R."
        assert result.lastName == "Tolkien"

    def test_last_comma_first_format(self):
        result = parse_author_name("Smith, John")
        assert result.firstName == "John"
        assert result.lastName == "Smith"

    def test_last_comma_first_with_middle(self):
        result = parse_author_name("Smith, John Michael")
        assert result.firstName == "John Michael"
        assert result.lastName == "Smith"

    def test_name_particles_von(self):
        result = parse_author_name("Ludwig van Beethoven")
        assert result.firstName == "Ludwig"
        assert result.lastName == "van Beethoven"

    def test_name_particles_de(self):
        result = parse_author_name("Charles de Gaulle")
        assert result.firstName == "Charles"
        assert result.lastName == "de Gaulle"

    def test_name_particles_van_der(self):
        result = parse_author_name("Johannes van der Waals")
        assert result.firstName == "Johannes"
        assert result.lastName == "van der Waals"

    def test_name_particles_multiple(self):
        result = parse_author_name("Maria de la Cruz")
        assert result.firstName == "Maria"
        assert result.lastName == "de la Cruz"

    def test_misplaced_initials_correction(self):
        # When middle initial incorrectly ends up looking like part of last name
        result = parse_author_name("Benjamin A. Black")
        assert result.lastName == "Black"
        assert "A." in result.firstName
        assert "Benjamin" in result.firstName

    def test_single_name_mononym(self):
        result = parse_author_name("Madonna")
        assert result.name == "Madonna"
        assert result.firstName is None
        assert result.lastName is None

    def test_single_name_organization(self):
        result = parse_author_name("WHO")
        assert result.name == "WHO"
        assert result.firstName is None
        assert result.lastName is None

    def test_empty_string(self):
        result = parse_author_name("")
        assert result.name == "Unknown"

    def test_whitespace_only(self):
        result = parse_author_name("   ")
        assert result.name == "Unknown"

    def test_to_zotero_creator_with_both_names(self):
        result = parse_author_name("John Smith")
        creator = result.to_zotero_creator()
        assert creator["firstName"] == "John"
        assert creator["lastName"] == "Smith"
        assert creator["creatorType"] == "author"

    def test_to_zotero_creator_single_name(self):
        result = parse_author_name("Madonna")
        creator = result.to_zotero_creator()
        assert creator["name"] == "Madonna"
        assert creator["creatorType"] == "author"

    def test_to_zotero_creator_custom_type(self):
        result = parse_author_name("John Smith")
        creator = result.to_zotero_creator("editor")
        assert creator["creatorType"] == "editor"


class TestNormalizeAuthorList:
    """Tests for normalize_author_list function."""

    def test_normalize_multiple_authors(self):
        authors = ["John Smith", "Jane Doe", "Robert Brown"]
        result = normalize_author_list(authors)
        assert len(result) == 3
        assert result[0].lastName == "Smith"
        assert result[1].lastName == "Doe"
        assert result[2].lastName == "Brown"

    def test_filters_empty_strings(self):
        authors = ["John Smith", "", "Jane Doe", "   "]
        result = normalize_author_list(authors)
        assert len(result) == 2

    def test_filters_none_values(self):
        authors = ["John Smith", None, "Jane Doe"]
        result = normalize_author_list(authors)
        assert len(result) == 2


class TestMetadataMerge:
    """Tests for merge_metadata_with_baseline function."""

    def test_baseline_preferred_for_priority_fields(self):
        baseline = {"date": "2023-05-15", "authors": ["John Smith"]}
        extracted = {"date": "sometime in 2023", "authors": ["J. Smith"]}

        result = merge_metadata_with_baseline(baseline, extracted)
        assert result["date"] == "2023-05-15"  # Baseline wins
        assert result["authors"] == ["John Smith"]  # Baseline wins

    def test_extracted_used_for_non_priority_fields(self):
        baseline = {"date": "2023", "title": "Original Title"}
        extracted = {"date": "2022", "title": "Better Title", "publisher": "Academic Press"}

        result = merge_metadata_with_baseline(baseline, extracted)
        assert result["date"] == "2023"  # Baseline wins (priority field)
        assert result["title"] == "Better Title"  # Extracted wins (not priority)
        assert result["publisher"] == "Academic Press"  # Extracted (baseline missing)

    def test_extracted_fills_gaps_in_baseline(self):
        baseline = {"authors": ["John Smith"]}
        extracted = {"authors": ["J. Smith"], "publisher": "Academic Press", "isbn": "123"}

        result = merge_metadata_with_baseline(baseline, extracted)
        assert result["authors"] == ["John Smith"]  # Baseline wins
        assert result["publisher"] == "Academic Press"  # Extracted fills gap
        assert result["isbn"] == "123"  # Extracted fills gap

    def test_empty_baseline_values_not_used(self):
        baseline = {"date": "", "authors": []}
        extracted = {"date": "2023", "authors": ["John Smith"]}

        result = merge_metadata_with_baseline(baseline, extracted)
        assert result["date"] == "2023"  # Extracted used (baseline empty)
        assert result["authors"] == ["John Smith"]  # Extracted used (baseline empty)

    def test_none_values_not_included(self):
        baseline = {"date": "2023"}
        extracted = {"date": None, "title": None, "publisher": "Press"}

        result = merge_metadata_with_baseline(baseline, extracted)
        assert result["date"] == "2023"
        assert "title" not in result  # None value not included
        assert result["publisher"] == "Press"

    def test_custom_priority_fields(self):
        baseline = {"title": "Original", "publisher": "Old Publisher"}
        extracted = {"title": "New", "publisher": "New Publisher"}

        # Make title a priority field
        result = merge_metadata_with_baseline(
            baseline, extracted, baseline_priority_fields={"title"}
        )
        assert result["title"] == "Original"  # Baseline wins (priority)
        assert result["publisher"] == "New Publisher"  # Extracted wins (not priority)

    def test_baseline_fills_remaining_gaps(self):
        baseline = {"date": "2023", "venue": "Nature", "abstract": "Some abstract"}
        extracted = {"date": "2022", "title": "My Paper"}

        result = merge_metadata_with_baseline(baseline, extracted)
        assert result["date"] == "2023"  # Baseline priority
        assert result["title"] == "My Paper"  # Extracted
        assert result["venue"] == "Nature"  # Baseline fills gap
        assert result["abstract"] == "Some abstract"  # Baseline fills gap
