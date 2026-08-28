"""Schemas for chunks, vectors and retrieval results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Payload keys that are always present on a stored chunk.
PAYLOAD_DOCUMENT_ID = "document_id"
PAYLOAD_CHUNK_ID = "chunk_id"
PAYLOAD_CATEGORY = "category"


class KnowledgeChunk(BaseModel):
    """One retrievable slice of a knowledge document.

    The payload stored alongside the vector is exactly this model, which keeps
    retrieval self-describing: a hit carries everything needed to cite it.
    """

    document_id: str = Field(description="Stable id of the parent document.")
    chunk_id: str = Field(description="Stable id of this chunk, unique per document.")
    chunk_index: int = Field(ge=0, description="Ordinal position within the document.")
    title: str = Field(description="Human-readable document title.")
    source: str = Field(description="Originating filename, used for citations.")
    category: str = Field(description="Knowledge area, e.g. ``lab_tests``.")
    section: str | None = Field(
        default=None, description="Nearest preceding heading, when one exists."
    )
    text: str = Field(description="The chunk body sent to the model as context.")

    def to_payload(self) -> dict[str, Any]:
        """Render the Qdrant payload for this chunk."""
        return self.model_dump()


class EmbeddedChunk(BaseModel):
    """A chunk paired with its embedding vector."""

    chunk: KnowledgeChunk
    vector: list[float]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchHit(BaseModel):
    """A single retrieval result."""

    text: str
    score: float = Field(description="Cosine similarity in [-1, 1]; higher is closer.")
    source: str
    title: str
    chunk_id: str
    document_id: str
    category: str | None = None
    section: str | None = None


class DocumentSummary(BaseModel):
    """Aggregate view of one indexed document."""

    document_id: str
    title: str
    source: str
    category: str
    chunk_count: int


class CollectionStats(BaseModel):
    """Vector-store counters, surfaced by the documents API."""

    collection: str
    exists: bool
    vector_count: int = 0
    document_count: int = 0
    dimension: int | None = None


class VectorStoreHealth(BaseModel):
    """Result of a vector-store reachability probe."""

    reachable: bool
    collection: str
    collection_exists: bool = False
    vector_count: int = 0
    detail: str | None = None
    checked_at: datetime | None = None


class IngestedDocument(BaseModel):
    """Result of ingesting one document."""

    document_id: str
    title: str
    source: str
    category: str
    chunks_written: int
    chunks_replaced: int = 0
    duration_ms: float = 0.0


class IngestResponse(BaseModel):
    """Payload returned by ``POST /documents/ingest``."""

    documents: list[IngestedDocument]
    failures: list[dict[str, str]] = Field(default_factory=list)
    total_chunks: int = 0
    duration_ms: float = 0.0


class DocumentListResponse(BaseModel):
    """Payload returned by ``GET /documents``."""

    collection: str
    document_count: int
    vector_count: int
    documents: list[DocumentSummary]


class DocumentDeleteResponse(BaseModel):
    """Payload returned by ``DELETE /documents/{document_id}``."""

    document_id: str
    chunks_deleted: int


class SearchRequest(BaseModel):
    """Request body for ``POST /rag/search``."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Natural-language question to search the knowledge base with.",
        examples=["What does a CBC test measure?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum chunks to return. Defaults to TOP_K.",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity. Chunks below this are dropped. "
            "Defaults to SCORE_THRESHOLD."
        ),
    )
    category: str | None = Field(
        default=None,
        description="Restrict to one knowledge area, e.g. ``lab_tests``.",
    )
    document_id: str | None = Field(
        default=None, description="Restrict to a single document."
    )
    source: str | None = Field(
        default=None, description="Restrict to a single source filename."
    )


class RetrievalTimings(BaseModel):
    """Latency breakdown for one retrieval, in milliseconds."""

    embed_ms: float
    search_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    """Ranked results for ``POST /rag/search``."""

    query: str
    results: list[SearchHit]
    retrieval_count: int = Field(description="Number of chunks returned.")
    top_k: int = Field(description="Limit actually applied.")
    score_threshold: float = Field(description="Threshold actually applied.")
    timings: RetrievalTimings
