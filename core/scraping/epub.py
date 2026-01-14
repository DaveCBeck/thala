"""EPUB processing - extract content and convert to markdown.

EPUB files are ZIP archives containing XHTML content. This module extracts
the text content and converts it to markdown format.
"""

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import html2text

logger = logging.getLogger(__name__)

# EPUB namespaces
NAMESPACES = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


class EpubProcessingError(Exception):
    """Error during EPUB processing."""

    pass


def is_epub_url(url: str) -> bool:
    """Check if URL points to an EPUB file."""
    clean_url = url.lower().split("?")[0].split("#")[0].rstrip("/")
    return clean_url.endswith(".epub")


def validate_epub_bytes(content: bytes) -> bool:
    """Validate that content is actually an EPUB.

    EPUB files are ZIP archives with specific structure.
    """
    # Check ZIP magic bytes
    if content[:4] != b"PK\x03\x04":
        return False

    # Check for EPUB structure
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            namelist = zf.namelist()
            # EPUB must have either META-INF/container.xml or mimetype
            has_container = "META-INF/container.xml" in namelist
            has_mimetype = "mimetype" in namelist

            if has_mimetype:
                mimetype = zf.read("mimetype").decode("utf-8", errors="ignore").strip()
                if "epub" in mimetype.lower():
                    return True

            return has_container
    except (zipfile.BadZipFile, Exception):
        return False


def _get_spine_items(zf: zipfile.ZipFile) -> list[str]:
    """Get ordered list of content files from EPUB spine.

    The spine defines the reading order of the EPUB content.
    """
    # Find the OPF file from container.xml
    try:
        container_xml = zf.read("META-INF/container.xml")
        container = ET.fromstring(container_xml)

        # Find rootfile path
        rootfile = container.find(
            ".//container:rootfile",
            NAMESPACES,
        )
        if rootfile is None:
            # Try without namespace
            rootfile = container.find(".//{*}rootfile")

        if rootfile is None:
            raise EpubProcessingError("Cannot find OPF file in container.xml")

        opf_path = rootfile.get("full-path")
        if not opf_path:
            raise EpubProcessingError("OPF file path not found")

    except KeyError:
        # No container.xml, try to find .opf file directly
        opf_files = [n for n in zf.namelist() if n.endswith(".opf")]
        if not opf_files:
            raise EpubProcessingError("Cannot find OPF file")
        opf_path = opf_files[0]

    # Parse OPF file
    opf_content = zf.read(opf_path)
    opf = ET.fromstring(opf_content)

    # Get base directory of OPF file for relative paths
    opf_dir = str(Path(opf_path).parent)
    if opf_dir == ".":
        opf_dir = ""

    # Build manifest id -> href mapping
    manifest = {}
    for item in opf.findall(".//{*}item"):
        item_id = item.get("id")
        href = item.get("href")
        media_type = item.get("media-type", "")
        if item_id and href:
            # Only include HTML/XHTML content
            if "html" in media_type or href.endswith((".html", ".xhtml", ".htm")):
                if opf_dir:
                    full_path = f"{opf_dir}/{href}"
                else:
                    full_path = href
                manifest[item_id] = full_path

    # Get spine order
    spine_items = []
    for itemref in opf.findall(".//{*}itemref"):
        idref = itemref.get("idref")
        if idref and idref in manifest:
            spine_items.append(manifest[idref])

    return spine_items


def _extract_text_from_html(html_content: bytes) -> str:
    """Extract text from HTML/XHTML content and convert to markdown."""
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.ignore_images = True
    h2t.ignore_emphasis = False
    h2t.body_width = 0  # Don't wrap lines

    try:
        text = html_content.decode("utf-8", errors="replace")
        return h2t.handle(text)
    except Exception as e:
        logger.warning(f"Failed to convert HTML: {e}")
        return ""


def process_epub_bytes(
    content: bytes,
    max_chapters: Optional[int] = None,
) -> str:
    """Convert EPUB bytes to markdown.

    Args:
        content: EPUB file content
        max_chapters: Maximum number of chapters to process (None = all)

    Returns:
        Markdown content extracted from EPUB

    Raises:
        EpubProcessingError: If EPUB processing fails
    """
    if not validate_epub_bytes(content):
        raise EpubProcessingError("Content is not a valid EPUB")

    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            # Get spine items in reading order
            spine_items = _get_spine_items(zf)

            if not spine_items:
                # Fallback: try to find any HTML files
                spine_items = [
                    n for n in zf.namelist()
                    if n.endswith((".html", ".xhtml", ".htm"))
                ]
                spine_items.sort()

            if not spine_items:
                raise EpubProcessingError("No content files found in EPUB")

            if max_chapters:
                spine_items = spine_items[:max_chapters]

            # Extract and convert each chapter
            markdown_parts = []
            for item_path in spine_items:
                try:
                    html_content = zf.read(item_path)
                    chapter_md = _extract_text_from_html(html_content)
                    if chapter_md.strip():
                        markdown_parts.append(chapter_md)
                except KeyError:
                    logger.debug(f"Could not read {item_path}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing {item_path}: {e}")
                    continue

            if not markdown_parts:
                raise EpubProcessingError("No readable content extracted from EPUB")

            markdown = "\n\n---\n\n".join(markdown_parts)
            logger.debug(f"Extracted {len(markdown)} chars from EPUB")
            return markdown.strip()

    except zipfile.BadZipFile as e:
        raise EpubProcessingError(f"Invalid EPUB (corrupt ZIP): {e}")
    except EpubProcessingError:
        raise
    except Exception as e:
        raise EpubProcessingError(f"EPUB processing failed: {e}")


def process_epub_file(
    file_path: str,
    max_chapters: Optional[int] = None,
) -> str:
    """Convert EPUB file to markdown.

    Args:
        file_path: Path to EPUB file
        max_chapters: Maximum number of chapters to process (None = all)

    Returns:
        Markdown content extracted from EPUB

    Raises:
        EpubProcessingError: If EPUB processing fails
    """
    path = Path(file_path)
    if not path.exists():
        raise EpubProcessingError(f"EPUB file not found: {file_path}")

    content = path.read_bytes()
    return process_epub_bytes(content, max_chapters=max_chapters)
