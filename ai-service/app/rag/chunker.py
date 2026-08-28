"""Section-aware chunking.

Splitting on a fixed character window cuts sentences and tables in half and
produces chunks that retrieve badly. This chunker instead:

1. splits the document on markdown headings, so a chunk never spans two
   unrelated sections;
2. packs whole paragraphs (and whole table rows) up to the size budget;
3. only falls back to sentence-level splitting for a single oversized block;
4. carries a sentence-aligned overlap between consecutive chunks so a fact
   sitting on a boundary appears in both.

Sizes are expressed in *approximate* tokens. A real tokenizer would need the
model to be loaded, which the ingest CLI must work without, so token counts are
estimated from whitespace-delimited words. The estimate runs slightly
conservative, which is the safe direction for a context budget.

Why the default budget is 220 and not 500
-----------------------------------------
``sentence-transformers/all-MiniLM-L6-v2`` has ``max_seq_length = 256`` word
pieces (verified against the loaded model). Anything beyond that is silently
truncated *before* the vector is computed, so a 500-token chunk would embed
only its first half while still being returned whole as context -- retrieval
would quietly ignore the tail of every long chunk. The default therefore sits
just under the model's real window. ``CHUNK_SIZE`` stays configurable for
models with a larger context.
"""

from __future__ import annotations

import re

from ..schemas.rag import KnowledgeChunk

#: English text averages ~0.75 words per token for common subword vocabularies.
_WORDS_PER_TOKEN = 0.75

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+(?=[A-Z(\[])")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

#: A tail smaller than this is not worth carrying as overlap.
_MIN_OVERLAP_TOKENS = 8
#: Chunks shorter than this are merged into their neighbour rather than stored.
_MIN_CHUNK_TOKENS = 20


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text``."""
    words = len(text.split())
    return max(1, round(words / _WORDS_PER_TOKEN)) if words else 0


class Section:
    """A heading and the body that follows it, before size-based splitting."""

    __slots__ = ("blocks", "heading")

    def __init__(self, heading: str | None) -> None:
        self.heading = heading
        self.blocks: list[str] = []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Section heading={self.heading!r} blocks={len(self.blocks)}>"


def split_sections(text: str) -> list[Section]:
    """Split markdown into sections keyed by their nearest heading.

    Consecutive table rows are kept together as one block so a table is never
    torn apart mid-row.
    """
    sections: list[Section] = []
    current = Section(None)
    paragraph: list[str] = []
    in_table = False

    def flush() -> None:
        nonlocal paragraph, in_table
        if paragraph:
            block = "\n".join(paragraph).strip()
            if block:
                current.blocks.append(block)
            paragraph = []
        in_table = False

    for line in text.split("\n"):
        heading_match = _HEADING.match(line)
        if heading_match:
            flush()
            if current.heading is not None or current.blocks:
                sections.append(current)
            current = Section(heading_match.group(2))
            continue

        is_table_row = bool(_TABLE_ROW.match(line))
        if not line.strip():
            # A blank line ends a paragraph, but not a table that continues.
            flush()
            continue

        if in_table and not is_table_row:
            flush()
        paragraph.append(line)
        in_table = is_table_row

    flush()
    if current.heading is not None or current.blocks:
        sections.append(current)

    return [section for section in sections if section.blocks]


def _split_oversized(block: str, budget: int) -> list[str]:
    """Break a single over-budget block on sentence boundaries."""
    sentences = _SENTENCE_END.split(block)
    if len(sentences) == 1:
        # No sentence boundaries (a long table or list): split on lines.
        sentences = [line for line in block.split("\n") if line.strip()]

    pieces: list[str] = []
    buffer: list[str] = []
    size = 0
    for sentence in sentences:
        cost = estimate_tokens(sentence)
        if buffer and size + cost > budget:
            pieces.append(" ".join(buffer))
            buffer, size = [], 0
        buffer.append(sentence)
        size += cost

    if buffer:
        pieces.append(" ".join(buffer))
    return pieces or [block]


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return the trailing ``overlap_tokens`` of ``text``, sentence-aligned."""
    if overlap_tokens < _MIN_OVERLAP_TOKENS:
        return ""

    sentences = _SENTENCE_END.split(text)
    tail: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        cost = estimate_tokens(sentence)
        if tail and size + cost > overlap_tokens:
            break
        tail.insert(0, sentence)
        size += cost

    joined = " ".join(tail).strip()
    # A tail equal to the whole chunk is not overlap, it is duplication.
    return "" if joined == text.strip() else joined


def chunk_text(
    text: str,
    *,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> list[tuple[str | None, str]]:
    """Chunk ``text`` into ``(section, body)`` pairs.

    Adjacent sections are packed together while they fit the budget, so a short
    heading does not become a 9-token chunk that retrieves noise. A section is
    only split internally when it exceeds the budget on its own.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Flatten to (heading, block) pairs, keeping the heading line in the body so
    # the embedding sees the topic the text belongs to.
    units: list[tuple[str | None, str]] = []
    for section in split_sections(text):
        blocks: list[str] = []
        # Split to the budget minus the overlap: a piece must still fit once
        # the previous chunk's tail is prepended to it.
        piece_budget = max(chunk_size - chunk_overlap, chunk_size // 2)
        for block in section.blocks:
            if estimate_tokens(block) > piece_budget:
                blocks.extend(_split_oversized(block, piece_budget))
            else:
                blocks.append(block)
        if not blocks:
            continue
        # The heading is prepended to the section's first block rather than
        # being a unit of its own -- otherwise a heading can be flushed alone
        # and become a 3-token chunk that retrieves nothing useful.
        if section.heading:
            blocks[0] = f"## {section.heading}\n\n{blocks[0]}"
        units.extend((section.heading, block) for block in blocks)

    results: list[tuple[str | None, str]] = []
    buffer: list[str] = []
    heading: str | None = None
    size = 0

    for block_heading, block in units:
        cost = estimate_tokens(block)
        if buffer and size + cost > chunk_size:
            body = "\n\n".join(buffer)
            results.append((heading, body))
            tail = _overlap_tail(body, chunk_overlap)
            # Dropping the overlap is better than exceeding the embedding
            # window, which would silently truncate the chunk's tail.
            if tail and estimate_tokens(tail) + cost > chunk_size:
                tail = ""
            buffer = [tail] if tail else []
            size = estimate_tokens(tail) if tail else 0
            heading = block_heading
        if not buffer:
            heading = block_heading
        buffer.append(block)
        size += cost

    if buffer:
        body = "\n\n".join(buffer).strip()
        if body:
            results.append((heading, body))

    return _merge_runts(results, chunk_size)


def _merge_runts(
    chunks: list[tuple[str | None, str]], chunk_size: int
) -> list[tuple[str | None, str]]:
    """Fold tiny trailing chunks into the previous chunk."""
    merged: list[tuple[str | None, str]] = []
    for heading, body in chunks:
        if (
            merged
            and estimate_tokens(body) < _MIN_CHUNK_TOKENS
            and estimate_tokens(merged[-1][1]) + estimate_tokens(body) <= chunk_size
        ):
            previous_heading, previous_body = merged.pop()
            merged.append((previous_heading, f"{previous_body}\n\n{body}"))
        else:
            merged.append((heading, body))
    return merged


def chunk_document(
    *,
    document_id: str,
    title: str,
    source: str,
    category: str,
    text: str,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> list[KnowledgeChunk]:
    """Chunk a loaded document into stored, citable :class:`KnowledgeChunk`s."""
    return [
        KnowledgeChunk(
            document_id=document_id,
            chunk_id=f"{document_id}::{index}",
            chunk_index=index,
            title=title,
            source=source,
            category=category,
            section=section,
            text=body,
        )
        for index, (section, body) in enumerate(
            chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    ]
