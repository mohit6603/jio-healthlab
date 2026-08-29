"""Document loading.

Turns a file (or an uploaded byte stream) into a :class:`RawDocument`: plain
text plus the metadata needed to cite it later. Format-specific parsing stops
here -- everything downstream works on normalised text.

Supported: ``.md``, ``.txt``, ``.pdf``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from ..core.errors import DocumentError
from ..core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".pdf"})

#: Files that describe the knowledge base rather than belonging to it.
_SKIPPED_NAMES = frozenset({"readme.md", "readme.txt", "index.md"})

#: Permissive MIME allow-list; the suffix is the authoritative check.
SUPPORTED_MIME_TYPES = frozenset(
    {
        "text/markdown",
        "text/x-markdown",
        "text/plain",
        "application/pdf",
        "application/octet-stream",  # browsers often send this for .md
    }
)

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
#: Underscores survive so a slug matches its category verbatim
#: (``lab_tests`` stays ``lab_tests``, not ``lab-tests``).
_SLUG_STRIP = re.compile(r"[^a-z0-9_]+")

#: Zero-width and formatting characters that survive NFKC normalisation.
_INVISIBLE = {ord(char): None for char in "\u200b\u200c\u200d\ufeff\u2060"}


@dataclass(slots=True)
class RawDocument:
    """A loaded document, before chunking."""

    document_id: str
    title: str
    source: str
    category: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ------------------------------------------------------------- front matter --
def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split simple ``key: value`` YAML front matter from the body.

    Deliberately hand-rolled: the front matter used here is flat scalars only,
    so a YAML dependency would buy nothing.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        metadata[key.strip().lower()] = value.strip().strip("'\"")

    return metadata, text[match.end() :]


# ------------------------------------------------------------ normalisation --
def normalise_text(text: str) -> str:
    """Canonicalise encoding, line endings and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # NFKC already folds NBSP into a plain space; zero-width characters
    # survive it and would otherwise end up inside chunk text.
    text = text.translate(_INVISIBLE)
    # Trailing whitespace, then runs of blank lines down to a single break.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(value: str) -> str:
    """Lowercase, hyphen-separated identifier safe for ids and filters."""
    slug = _SLUG_STRIP.sub("-", value.strip().lower()).strip("-")
    return slug or "document"


# ------------------------------------------------------------------ readers --
def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back rather than reject: exported notes are often latin-1.
        logger.warning("document_decoded_with_fallback", extra={"source": path.name})
        return path.read_text(encoding="latin-1")
    except OSError as exc:
        raise DocumentError(f"Could not read '{path.name}'.") from exc


def extract_pdf_text(data: bytes, source: str) -> str:
    """Extract text from a PDF, page by page.

    Scanned PDFs contain no text layer; those come back effectively empty and
    are rejected upstream rather than silently indexed as blank.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise DocumentError("PDF support requires the 'pypdf' package.") from exc

    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    # pypdf raises a wide range of types for malformed files.
    except Exception as exc:
        raise DocumentError(f"Could not parse PDF '{source}'.") from exc

    return "\n\n".join(page.strip() for page in pages if page.strip())


# ------------------------------------------------------------------ loading --
def _derive_title(metadata: dict[str, str], text: str, fallback: str) -> str:
    if title := metadata.get("title"):
        return title
    if match := _H1.search(text):
        return match.group(1)
    return fallback.replace("_", " ").replace("-", " ").title()


def load_bytes(
    data: bytes,
    filename: str,
    *,
    category: str | None = None,
    title: str | None = None,
    document_id: str | None = None,
) -> RawDocument:
    """Build a :class:`RawDocument` from raw bytes (an upload)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            f"Unsupported file type '{suffix or filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}.",
            code="UNSUPPORTED_FILE_TYPE",
            details={"supported": sorted(SUPPORTED_SUFFIXES)},
        )

    if suffix == ".pdf":
        body = extract_pdf_text(data, filename)
        metadata: dict[str, str] = {}
    else:
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError:
            decoded = data.decode("latin-1", errors="replace")
        metadata, decoded = parse_front_matter(decoded)
        body = decoded

    body = normalise_text(body)
    if not body:
        raise DocumentError(
            f"'{filename}' contains no extractable text. "
            "Scanned PDFs need OCR before they can be indexed.",
            code="EMPTY_DOCUMENT",
        )

    stem = Path(filename).stem
    resolved_category = category or metadata.get("category") or "uploaded"

    return RawDocument(
        document_id=document_id or f"{slugify(resolved_category)}/{slugify(stem)}",
        title=title or _derive_title(metadata, body, stem),
        source=Path(filename).name,
        category=resolved_category,
        text=body,
        metadata=metadata,
    )


def load_file(path: Path, *, root: Path | None = None) -> RawDocument:
    """Load one file from disk.

    ``root`` gives the document its category: a file at
    ``knowledge/lab_tests/cbc.md`` becomes category ``lab_tests``.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            f"Unsupported file type '{suffix}' for '{path.name}'.",
            code="UNSUPPORTED_FILE_TYPE",
        )

    data = path.read_bytes() if suffix == ".pdf" else _read_text(path).encode("utf-8")

    category = None
    if root is not None:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = None
        if relative is not None and len(relative.parts) > 1:
            category = relative.parts[0]

    return load_bytes(data, path.name, category=category)


def discover(root: Path) -> list[Path]:
    """List every ingestible file under ``root``, sorted for reproducibility."""
    if not root.exists():
        raise DocumentError(
            f"Knowledge directory '{root}' does not exist.",
            code="KNOWLEDGE_DIR_MISSING",
        )

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.name.lower() not in _SKIPPED_NAMES
        and not path.name.startswith(".")
    )
