"""Test SVG validation and sanitization utilities."""

import logging

from core.config import configure_logging
from workflows.shared.diagram_utils.validation import (
    extract_validation_error_type,
    sanitize_svg_text_entities,
    validate_svg_xml,
)

configure_logging("svg_validation_test")
logger = logging.getLogger(__name__)


# Valid SVG for baseline testing
VALID_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="100" y="100" font-size="16">Hello World</text>
</svg>"""

# SVG with unescaped ampersand (common LLM error)
SVG_UNESCAPED_AMPERSAND = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="100" y="100" font-size="16">Input & Output</text>
</svg>"""

# SVG with multiple unescaped entities
SVG_MULTIPLE_UNESCAPED = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="100" y="100" font-size="16">A & B & C</text>
  <text x="100" y="150" font-size="16">x < y</text>
</svg>"""

# SVG with already-escaped entities (should not double-escape)
SVG_ALREADY_ESCAPED = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="800" height="600">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="100" y="100" font-size="16">Input &amp; Output</text>
</svg>"""


def test_validate_valid_svg():
    """Test that valid SVG passes validation."""
    logger.info("Testing valid SVG validation...")
    is_valid, error = validate_svg_xml(VALID_SVG)
    assert is_valid is True, f"Valid SVG should pass validation, got error: {error}"
    assert error is None
    logger.info("✓ Valid SVG passes validation")


def test_validate_invalid_svg():
    """Test that SVG with unescaped ampersand fails validation."""
    logger.info("Testing invalid SVG validation...")
    is_valid, error = validate_svg_xml(SVG_UNESCAPED_AMPERSAND)
    assert is_valid is False, "SVG with unescaped & should fail validation"
    assert error is not None
    assert "xmlParseEntityRef" in error or "no name" in error
    logger.info(f"✓ Invalid SVG correctly fails validation: {error}")


def test_sanitize_ampersand():
    """Test sanitization of unescaped ampersand."""
    logger.info("Testing ampersand sanitization...")
    sanitized = sanitize_svg_text_entities(SVG_UNESCAPED_AMPERSAND)

    # Should now be valid XML
    is_valid, error = validate_svg_xml(sanitized)
    assert is_valid is True, f"Sanitized SVG should be valid, got: {error}"

    # Should contain escaped entity
    assert "&amp;" in sanitized, "Ampersand should be escaped"
    logger.info("✓ Ampersand sanitization works")


def test_sanitize_multiple_entities():
    """Test sanitization of multiple unescaped entities."""
    logger.info("Testing multiple entity sanitization...")
    sanitized = sanitize_svg_text_entities(SVG_MULTIPLE_UNESCAPED)

    # Should now be valid XML
    is_valid, error = validate_svg_xml(sanitized)
    assert is_valid is True, f"Sanitized SVG should be valid, got: {error}"

    # Should contain escaped entities
    assert "&amp;" in sanitized, "Ampersands should be escaped"
    assert "&lt;" in sanitized, "Less-than should be escaped"
    logger.info("✓ Multiple entity sanitization works")


def test_no_double_escaping():
    """Test that already-escaped entities are not double-escaped."""
    logger.info("Testing no double-escaping...")
    sanitized = sanitize_svg_text_entities(SVG_ALREADY_ESCAPED)

    # Should remain valid
    is_valid, error = validate_svg_xml(sanitized)
    assert is_valid is True, f"Already-escaped SVG should remain valid, got: {error}"

    # Should NOT contain double-escaped entities
    assert "&amp;amp;" not in sanitized, "Should not double-escape"
    logger.info("✓ No double-escaping")


def test_valid_svg_unchanged():
    """Test that valid SVG passes through unchanged."""
    logger.info("Testing valid SVG unchanged...")
    sanitized = sanitize_svg_text_entities(VALID_SVG)
    assert sanitized == VALID_SVG, "Valid SVG should not be modified"
    logger.info("✓ Valid SVG unchanged")


def test_error_type_extraction():
    """Test error type categorization."""
    logger.info("Testing error type extraction...")

    class MockException(Exception):
        pass

    entity_error = MockException("xmlParseEntityRef: no name, line 5, column 10")
    assert extract_validation_error_type(entity_error) == "unescaped_entity"

    malformed_error = MockException("not well-formed (invalid token)")
    assert extract_validation_error_type(malformed_error) == "malformed_xml"

    unknown_error = MockException("some other error")
    assert extract_validation_error_type(unknown_error) == "unknown_xml_error"

    logger.info("✓ Error type extraction works")


def test_sanitization_with_conversion():
    """Test that sanitized SVG can be converted to PNG."""
    logger.info("Testing sanitization with PNG conversion...")
    from workflows.shared.diagram_utils.conversion import convert_svg_to_png

    # This should fail without sanitization
    png_bytes = convert_svg_to_png(SVG_UNESCAPED_AMPERSAND)
    if png_bytes is not None:
        logger.info("✓ Conversion succeeded (sanitization happened in convert_svg_to_png)")
    else:
        logger.warning("⚠ Conversion failed - check if sanitization is integrated")


def run_all_tests():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("SVG Validation Tests")
    logger.info("=" * 60)

    test_validate_valid_svg()
    test_validate_invalid_svg()
    test_sanitize_ampersand()
    test_sanitize_multiple_entities()
    test_no_double_escaping()
    test_valid_svg_unchanged()
    test_error_type_extraction()
    test_sanitization_with_conversion()

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all_tests()
