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
