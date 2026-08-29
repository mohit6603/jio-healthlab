"""Chunker tests: sectioning, budget adherence, overlap and table safety."""

import pytest

from app.rag.chunker import (
    chunk_document,
    chunk_text,
    estimate_tokens,
    split_sections,
)

DOC = """# Complete Blood Count

## What a CBC measures

A CBC measures red blood cells, white blood cells and platelets.

## Sample requirements

A CBC uses an EDTA tube. Fasting is not required.

## Turnaround

Routine results are available within four hours.
"""


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") > 0
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)


def test_split_sections_uses_headings():
    sections = split_sections(DOC)

    headings = [section.heading for section in sections]
    assert "What a CBC measures" in headings
    assert "Sample requirements" in headings


def test_split_sections_keeps_table_rows_together():
    text = "## Panel\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"

    blocks = split_sections(text)[0].blocks

    table_blocks = [block for block in blocks if "|" in block]
    assert len(table_blocks) == 1
    assert table_blocks[0].count("\n") == 3


def test_chunks_stay_within_the_budget():
    chunks = chunk_text(DOC, chunk_size=60, chunk_overlap=10)

    assert chunks
    # A little slack: the last block added is allowed to cross the budget.
    assert all(estimate_tokens(body) <= 60 * 2 for _, body in chunks)


def test_heading_text_is_kept_in_the_body():
    chunks = chunk_text(DOC, chunk_size=40, chunk_overlap=5)

    assert any("## What a CBC measures" in body for _, body in chunks)


def test_section_metadata_is_attached():
    chunks = chunk_text(DOC, chunk_size=40, chunk_overlap=5)

    assert {section for section, _ in chunks} & {
        "What a CBC measures",
        "Sample requirements",
        "Turnaround",
    }


def test_small_document_becomes_one_chunk():
    chunks = chunk_text(DOC, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1


def test_overlap_repeats_boundary_text():
    text = "## S\n\n" + " ".join(f"Sentence number {i} here." for i in range(60))

    chunks = chunk_text(text, chunk_size=40, chunk_overlap=15)

    assert len(chunks) > 1
    first_tail_words = set(chunks[0][1].split()[-8:])
    assert first_tail_words & set(chunks[1][1].split())


def test_zero_overlap_is_allowed():
    chunks = chunk_text(DOC, chunk_size=40, chunk_overlap=0)

    assert len(chunks) >= 1


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text(DOC, chunk_size=50, chunk_overlap=50)


def test_oversized_block_is_split_on_sentences():
    long_block = " ".join(f"This is sentence {i}." for i in range(200))

    chunks = chunk_text(f"## S\n\n{long_block}", chunk_size=50, chunk_overlap=5)

    assert len(chunks) > 3


def test_no_runt_chunks_are_emitted():
    chunks = chunk_text(DOC, chunk_size=45, chunk_overlap=8)

    # The final chunk may legitimately be short; earlier ones should not be.
    assert all(estimate_tokens(body) >= 15 for _, body in chunks[:-1])


def test_empty_text_produces_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


# ------------------------------------------------------------- documents ---
def test_chunk_document_builds_stable_ids():
    chunks = chunk_document(
        document_id="lab_tests/cbc",
        title="CBC Guide",
        source="cbc.md",
        category="lab_tests",
        text=DOC,
        chunk_size=40,
        chunk_overlap=5,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        f"lab_tests/cbc::{index}" for index in range(len(chunks))
    ]


def test_chunk_document_propagates_citation_metadata():
    chunks = chunk_document(
        document_id="lab_tests/cbc",
        title="CBC Guide",
        source="cbc.md",
        category="lab_tests",
        text=DOC,
    )

    assert all(chunk.title == "CBC Guide" for chunk in chunks)
    assert all(chunk.source == "cbc.md" for chunk in chunks)
    assert all(chunk.category == "lab_tests" for chunk in chunks)


def test_chunk_indexes_are_contiguous():
    chunks = chunk_document(
        document_id="d",
        title="T",
        source="s.md",
        category="c",
        text=DOC,
        chunk_size=40,
        chunk_overlap=5,
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


# --------------------------------------------------- real knowledge base ---
def test_bundled_documents_fit_the_embedding_window(settings):
    """Every produced chunk must fit MiniLM's 256 word-piece window."""
    from app.rag.loaders import discover, load_file

    for path in discover(settings.knowledge_path):
        document = load_file(path, root=settings.knowledge_path)
        chunks = chunk_document(
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            category=document.category,
            text=document.text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        assert chunks, f"{path.name} produced no chunks"
        for chunk in chunks:
            assert estimate_tokens(chunk.text) <= settings.embedding_max_tokens, (
                f"{chunk.chunk_id} exceeds the embedding window"
            )
