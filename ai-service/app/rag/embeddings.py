"""Embedding abstraction.

Retrieval depends on this interface, never on a concrete model class, so the
embedding backend can be swapped (a different sentence-transformer, a hosted
embedding API) without touching ingestion or retrieval.

The concrete Sentence-Transformers implementation is added in the embedding
phase; this module defines the contract and the shared text-preparation rules
that any implementation must follow.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..schemas.rag import KnowledgeChunk

if TYPE_CHECKING:
    from ..config import Settings


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into vectors."""

    @property
    def dimension(self) -> int:
        """Width of the vectors this embedder produces."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model, recorded with ingested chunks."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string -- used for queries."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings -- used for ingestion."""
        ...


def chunk_embedding_input(chunk: KnowledgeChunk) -> str:
    """Build the text actually handed to the embedder for a chunk.

    The stored ``text`` stays verbatim so citations quote the source exactly,
    but the *embedding* input is prefixed with the document title and section.
    A chunk reading "Fasting is not required." is nearly meaningless on its
    own; prefixed with "Complete Blood Count (CBC) Guide -- Sample
    requirements" it lands in the right region of the vector space.
    """
    parts = [chunk.title]
    if chunk.section and chunk.section.lower() not in chunk.title.lower():
        parts.append(chunk.section)
    heading = " — ".join(parts)
    return f"{heading}\n\n{chunk.text}"


class SentenceTransformerEmbedder:
    """Sentence-Transformers embedder with lazy, one-time model loading.

    The model is loaded on first use rather than at import or startup: the
    container must become healthy without waiting on a download, and a process
    that only serves ``/health`` should never pay for the weights.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        from ..config import get_settings

        self._settings = settings or get_settings()
        self._model: object | None = None

    # ---------------------------------------------------------- metadata ---
    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    @property
    def dimension(self) -> int:
        return self._settings.embedding_dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------ loading --
    def _load(self):
        """Load the model once, recording success or failure for /health."""
        if self._model is not None:
            return self._model

        from ..core import runtime
        from ..core.errors import ModelUnavailableError
        from ..core.logging import get_logger
        from ..utils.optional import require

        logger = get_logger(__name__)
        require("sentence_transformers", "Embedding")

        started = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self._settings.embedding_model,
                device=self._settings.embedding_device,
            )
        # Any load failure (network, disk, OOM) must be reported, not raised raw.
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            runtime.mark_failed(runtime.EMBEDDING, reason)
            logger.error(
                "embedding_model_load_failed",
                extra={"model": self._settings.embedding_model},
            )
            raise ModelUnavailableError(
                f"Could not load embedding model "
                f"'{self._settings.embedding_model}'.",
                details={"reason": reason},
            ) from exc

        # A dimension mismatch would only show up later as a Qdrant rejection.
        # sentence-transformers 6 renamed this accessor; support both.
        read_dimension = getattr(
            model, "get_embedding_dimension", None
        ) or model.get_sentence_embedding_dimension
        actual = int(read_dimension())
        if actual != self._settings.embedding_dimension:
            raise ModelUnavailableError(
                f"Model '{self._settings.embedding_model}' emits {actual}-d "
                f"vectors but EMBEDDING_DIMENSION is "
                f"{self._settings.embedding_dimension}.",
                code="EMBEDDING_DIMENSION_MISMATCH",
            )

        self._model = model
        runtime.mark_loaded(runtime.EMBEDDING)
        logger.info(
            "embedding_model_loaded",
            extra={
                "model": self._settings.embedding_model,
                "dimension": actual,
                "device": self._settings.embedding_device,
                "load_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return model

    def warm_up(self) -> None:
        """Force the model to load now (used by the ingest CLI)."""
        self._load()

    # --------------------------------------------------------- embedding ---
    def embed_text(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load()
        vectors = model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self._settings.embedding_batch_size,
            # Cosine distance in Qdrant assumes unit-length vectors.
            # all-MiniLM-L6-v2 already ends its pipeline with a Normalize
            # module, but passing the flag makes unit length a property of
            # this embedder rather than of whichever model is configured.
            normalize_embeddings=self._settings.embedding_normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


_embedder: SentenceTransformerEmbedder | None = None


def get_embedder() -> SentenceTransformerEmbedder:
    """Process-wide singleton -- the model is loaded at most once."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def reset_embedder() -> None:
    """Drop the singleton (test helper)."""
    global _embedder
    _embedder = None
