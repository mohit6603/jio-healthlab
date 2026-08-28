"""Loader tests: front matter, normalisation, discovery and rejection."""

import pytest

from app.core.errors import DocumentError
from app.rag.loaders import (
    discover,
    load_bytes,
    load_file,
    normalise_text,
    parse_front_matter,
    slugify,
)


# ------------------------------------------------------------ front matter --
def test_front_matter_is_parsed_and_stripped():
    metadata, body = parse_front_matter(
        "---\ntitle: CBC Guide\ncategory: lab_tests\n---\n# Heading\n\nBody."
    )

    assert metadata == {"title": "CBC Guide", "category": "lab_tests"}
    assert body.startswith("# Heading")


def test_document_without_front_matter_is_untouched():
    metadata, body = parse_front_matter("# Heading\n\nBody.")

    assert metadata == {}
    assert body == "# Heading\n\nBody."


def test_front_matter_quotes_are_stripped():
    metadata, _ = parse_front_matter("---\ntitle: 'Quoted Title'\n---\nBody")

    assert metadata["title"] == "Quoted Title"


# ----------------------------------------------------------- normalisation --
def test_normalise_collapses_blank_line_runs():
    assert normalise_text("a\n\n\n\n\nb") == "a\n\nb"


def test_normalise_converts_line_endings():
    assert normalise_text("a\r\nb\rc") == "a\nb\nc"


def test_normalise_strips_zero_width_characters():
    assert normalise_text("he​llo﻿") == "hello"


def test_normalise_strips_trailing_whitespace():
    assert normalise_text("line   \n  next  ") == "line\n  next"


def test_slugify_preserves_underscores():
    assert slugify("Lab_Tests Guide!") == "lab_tests-guide"


# ------------------------------------------------------------------ loading --
def test_load_markdown_derives_title_from_front_matter():
    document = load_bytes(
        b"---\ntitle: Thyroid Guide\ncategory: lab_tests\n---\n# Other\n\nBody text.",
        "thyroid.md",
    )

    assert document.title == "Thyroid Guide"
    assert document.category == "lab_tests"
    assert document.source == "thyroid.md"


def test_load_markdown_falls_back_to_h1():
    document = load_bytes(b"# Kidney Function\n\nBody.", "kidney.md")

    assert document.title == "Kidney Function"


def test_load_markdown_falls_back_to_filename():
    document = load_bytes(b"Just body text.", "sample_collection.txt")

    assert document.title == "Sample Collection"


def test_explicit_category_wins_over_front_matter():
    document = load_bytes(
        b"---\ncategory: lab_tests\n---\nBody.", "x.md", category="faq"
    )

    assert document.category == "faq"


def test_document_id_combines_category_and_stem():
    document = load_bytes(b"Body.", "cbc.md", category="lab_tests")

    assert document.document_id == "lab_tests/cbc"


def test_unsupported_extension_is_rejected():
    with pytest.raises(DocumentError) as excinfo:
        load_bytes(b"data", "notes.docx")

    assert excinfo.value.code == "UNSUPPORTED_FILE_TYPE"


def test_empty_document_is_rejected():
    with pytest.raises(DocumentError) as excinfo:
        load_bytes(b"   \n\n  ", "blank.md")

    assert excinfo.value.code == "EMPTY_DOCUMENT"


def test_non_utf8_bytes_are_decoded_with_fallback():
    document = load_bytes("Café résumé".encode("latin-1"), "notes.txt")

    assert "Caf" in document.text


# ---------------------------------------------------------------- discovery --
def test_discover_finds_bundled_knowledge(settings):
    paths = discover(settings.knowledge_path)

    assert len(paths) >= 12
    assert all(path.suffix in {".md", ".txt", ".pdf"} for path in paths)


def test_discover_excludes_readme(settings):
    names = {path.name.lower() for path in discover(settings.knowledge_path)}

    assert "readme.md" not in names


def test_discover_is_sorted(settings):
    paths = discover(settings.knowledge_path)

    assert paths == sorted(paths)


def test_discover_missing_directory_raises(tmp_path):
    with pytest.raises(DocumentError) as excinfo:
        discover(tmp_path / "nope")

    assert excinfo.value.code == "KNOWLEDGE_DIR_MISSING"


def test_load_file_infers_category_from_directory(settings, tmp_path):
    folder = tmp_path / "sop"
    folder.mkdir()
    path = folder / "workflow.md"
    path.write_text("# Workflow\n\nBody.", encoding="utf-8")

    document = load_file(path, root=tmp_path)

    assert document.category == "sop"
    assert document.document_id == "sop/workflow"


def test_bundled_cbc_document_loads(settings):
    path = settings.knowledge_path / "lab_tests" / "cbc.md"

    document = load_file(path, root=settings.knowledge_path)

    assert document.category == "lab_tests"
    assert "red blood cells" in document.text.lower()
    assert document.metadata["title"].startswith("Complete Blood Count")
