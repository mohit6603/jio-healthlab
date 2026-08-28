"""Ingestion pipeline tests.

A deterministic stub embedder stands in for the real model: these tests assert
pipeline mechanics (chunking, replacement, batching, error handling), not
embedding quality.
"""

import hashlib

import pytest

from app.core.errors import DocumentError
from app.rag.embeddings import chunk_embedding_input
from app.rag.loaders import RawDocument
from app.rag.pipeline import IngestionPipeline
from app.rag.vector_store import VectorStore
from app.schemas.rag import KnowledgeChunk
from tests.test_vector_store import StubClient

DIMENSION = 384


class StubEmbedder:
    """Deterministic, dependency-free embedder for pipeline tests."""

    model_name = "stub-embedder"
    dimension = DIMENSION

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [digest[i % len(digest)] / 255.0 for i in range(self.dimension)]
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            vectors.append([value / norm for value in raw])
        return vectors


@pytest.fixture(name="pipeline_parts")
def pipeline_parts_fixture(settings):
    client = StubClient()
    store = VectorStore(settings=settings, client=client)
    embedder = StubEmbedder()
    return IngestionPipeline(embedder, store, settings), client, embedder


def make_document(text: str = "# Title\n\n## Section\n\nBody text here.") -> RawDocument:
    return RawDocument(
        document_id="lab_tests/cbc",
        title="CBC Guide",
        source="cbc.md",
        category="lab_tests",
        text=text,
    )


# ------------------------------------------------------------ embed input --
def test_embedding_input_prefixes_title_and_section():
    chunk = KnowledgeChunk(
        document_id="d",
        chunk_id="d::0",
        chunk_index=0,
        title="CBC Guide",
        source="cbc.md",
        category="lab_tests",
        section="Sample requirements",
        text="Fasting is not required.",
    )

    prepared = chunk_embedding_input(chunk)

    assert prepared.startswith("CBC Guide — Sample requirements")
    assert "Fasting is not required." in prepared


def test_embedding_input_omits_redundant_section():
    chunk = KnowledgeChunk(
        document_id="d",
        chunk_id="d::0",
        chunk_index=0,
        title="CBC Guide",
        source="cbc.md",
        category="lab_tests",
        section="cbc guide",
        text="Body.",
    )

    assert chunk_embedding_input(chunk).count("CBC Guide") == 1


# --------------------------------------------------------------- ingestion --
def test_ingest_document_writes_chunks(pipeline_parts):
    pipeline, client, _ = pipeline_parts

    result = pipeline.ingest_document(make_document())

    assert result.chunks_written == len(client.upserted)
    assert result.chunks_written > 0
    assert result.document_id == "lab_tests/cbc"


def test_ingest_embeds_in_a_single_batch(pipeline_parts):
    pipeline, client, embedder = pipeline_parts

    pipeline.ingest_document(make_document())

    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == len(client.upserted)


def test_ingest_embeds_prefixed_text_not_raw_text(pipeline_parts):
    pipeline, _, embedder = pipeline_parts

    pipeline.ingest_document(make_document())

    assert embedder.calls[0][0].startswith("CBC Guide")


def test_ingest_deletes_previous_chunks_first(pipeline_parts):
    """Re-ingesting a shortened document must not orphan its old chunks."""
    pipeline, client, _ = pipeline_parts
    client.existing = True
    client.count_value = 9

    result = pipeline.ingest_document(make_document())

    assert result.chunks_replaced == 9
    assert client.deleted


def test_ingest_rejects_empty_document(pipeline_parts):
    pipeline, _, _ = pipeline_parts

    with pytest.raises(DocumentError) as excinfo:
        pipeline.ingest_document(make_document(text="   "))

    assert excinfo.value.code == "EMPTY_DOCUMENT"


def test_ingest_detects_dimension_mismatch(settings):
    """A wrong-width embedder must fail loudly, not corrupt the collection."""
    store = VectorStore(settings=settings, client=StubClient())
    pipeline = IngestionPipeline(StubEmbedder(dimension=128), store, settings)

    with pytest.raises(DocumentError) as excinfo:
        pipeline.ingest_document(make_document())

    assert excinfo.value.code == "EMBEDDING_DIMENSION_MISMATCH"


def test_ingest_detects_vector_count_mismatch(settings):
    class ShortEmbedder(StubEmbedder):
        def embed_documents(self, texts):
            return super().embed_documents(texts)[:-1]

    store = VectorStore(settings=settings, client=StubClient())
    pipeline = IngestionPipeline(ShortEmbedder(), store, settings)

    with pytest.raises(DocumentError) as excinfo:
        pipeline.ingest_document(
            make_document(
                text="## A\n\n" + "\n\n".join(f"Para {i} body." for i in range(40))
            )
        )

    assert excinfo.value.code == "EMBEDDING_MISMATCH"


def test_ingest_bytes_enforces_the_size_limit(settings):
    settings.max_upload_bytes = 32
    store = VectorStore(settings=settings, client=StubClient())
    pipeline = IngestionPipeline(StubEmbedder(), store, settings)

    with pytest.raises(DocumentError) as excinfo:
        pipeline.ingest_bytes(b"x" * 100, "big.md")

    assert excinfo.value.code == "DOCUMENT_TOO_LARGE"


# --------------------------------------------------------------- directory --
def test_ingest_directory_covers_the_knowledge_base(pipeline_parts, settings):
    pipeline, _, _ = pipeline_parts

    report = pipeline.ingest_directory(settings.knowledge_path)

    assert report.document_count == 12
    assert report.chunk_count > 40
    assert report.failures == []


def test_ingest_directory_records_failures_without_aborting(
    pipeline_parts, settings, tmp_path
):
    pipeline, _, _ = pipeline_parts
    (tmp_path / "good.md").write_text("# Good\n\nUsable body text.", encoding="utf-8")
    (tmp_path / "blank.md").write_text("   \n  ", encoding="utf-8")

    report = pipeline.ingest_directory(tmp_path)

    assert report.document_count == 1
    assert len(report.failures) == 1
    assert report.failures[0][0] == "blank.md"


def test_ingest_directory_reports_duration(pipeline_parts, settings):
    pipeline, _, _ = pipeline_parts

    report = pipeline.ingest_directory(settings.knowledge_path)

    assert report.duration_ms >= 0
