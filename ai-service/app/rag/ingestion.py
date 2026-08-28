"""Document ingestion pipeline.

    Document -> load -> normalise -> clean -> chunk -> embed -> store

Every stage is a plain function or an injected collaborator, so the pipeline is
testable end to end without a model or a running Qdrant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from ..core.errors import DocumentError
from ..core.logging import get_logger
from ..schemas.rag import EmbeddedChunk, KnowledgeChunk
from .chunker import chunk_document
from .embeddings import Embedder, chunk_embedding_input
from .loaders import RawDocument, discover, load_bytes, load_file
from .vector_store import VectorStore

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestionResult:
    """Outcome of ingesting a single document."""

    document_id: str
    title: str
    source: str
    category: str
    chunks_written: int
    chunks_replaced: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class IngestionReport:
    """Outcome of a directory-wide ingestion run."""

    documents: list[IngestionResult] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return sum(item.chunks_written for item in self.documents)


class IngestionPipeline:
    """Loads, chunks, embeds and stores knowledge documents."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._settings = settings or get_settings()

    # ------------------------------------------------------------ single ---
    def ingest_document(
        self, document: RawDocument, *, collection: str | None = None
    ) -> IngestionResult:
        """Run one loaded document through chunk -> embed -> store."""
        started = time.perf_counter()

        chunks = chunk_document(
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            category=document.category,
            text=document.text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise DocumentError(
                f"'{document.source}' produced no chunks.", code="EMPTY_DOCUMENT"
            )

        # Re-ingesting a shortened document must not leave orphaned chunks
        # behind from the previous, longer version.
        replaced = self._store.delete_document(
            document.document_id, collection=collection
        )

        embedded = self._embed(chunks)
        written = self._store.upsert_chunks(embedded, collection=collection)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "document_ingested",
            extra={
                "document_id": document.document_id,
                "source": document.source,
                "category": document.category,
                "chunks_written": written,
                "chunks_replaced": replaced,
                "duration_ms": duration_ms,
            },
        )

        return IngestionResult(
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            category=document.category,
            chunks_written=written,
            chunks_replaced=replaced,
            duration_ms=duration_ms,
        )

    def _embed(self, chunks: list[KnowledgeChunk]) -> list[EmbeddedChunk]:
        """Embed chunks in one batched call."""
        inputs = [chunk_embedding_input(chunk) for chunk in chunks]
        vectors = self._embedder.embed_documents(inputs)

        if len(vectors) != len(chunks):
            raise DocumentError(
                "Embedder returned "
                f"{len(vectors)} vectors for {len(chunks)} chunks.",
                code="EMBEDDING_MISMATCH",
            )

        expected = self._settings.embedding_dimension
        if vectors and len(vectors[0]) != expected:
            raise DocumentError(
                f"Embedder produced {len(vectors[0])}-d vectors but the "
                f"collection expects {expected}-d. Check EMBEDDING_MODEL and "
                "EMBEDDING_DIMENSION agree, and re-index after changing them.",
                code="EMBEDDING_DIMENSION_MISMATCH",
            )

        return [
            EmbeddedChunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    # ------------------------------------------------------------ sources ---
    def ingest_file(
        self, path: Path, *, root: Path | None = None, collection: str | None = None
    ) -> IngestionResult:
        return self.ingest_document(load_file(path, root=root), collection=collection)

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        category: str | None = None,
        title: str | None = None,
        collection: str | None = None,
    ) -> IngestionResult:
        max_bytes = self._settings.max_upload_bytes
        if len(data) > max_bytes:
            raise DocumentError(
                f"'{filename}' is {len(data)} bytes; the limit is {max_bytes}.",
                code="DOCUMENT_TOO_LARGE",
                details={"max_bytes": max_bytes},
            )

        document = load_bytes(data, filename, category=category, title=title)
        return self.ingest_document(document, collection=collection)

    def ingest_directory(
        self, root: Path | None = None, *, collection: str | None = None
    ) -> IngestionReport:
        """Ingest every supported file under ``root``.

        One bad document does not abort the run -- it is recorded as a failure
        so a single malformed file cannot block the whole knowledge base.
        """
        root = root or self._settings.knowledge_path
        started = time.perf_counter()
        report = IngestionReport()

        for path in discover(root):
            try:
                report.documents.append(
                    self.ingest_file(path, root=root, collection=collection)
                )
            except DocumentError as exc:
                logger.warning(
                    "document_ingest_failed",
                    extra={"source": path.name, "code": exc.code},
                )
                report.failures.append((path.name, exc.message))

        report.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "ingestion_run_completed",
            extra={
                "documents": report.document_count,
                "chunks": report.chunk_count,
                "failures": len(report.failures),
                "duration_ms": report.duration_ms,
            },
        )
        return report

    # ------------------------------------------------------------ removal ---
    def delete_document(
        self, document_id: str, *, collection: str | None = None
    ) -> int:
        return self._store.delete_document(document_id, collection=collection)
