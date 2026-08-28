"""Knowledge-base document lifecycle endpoints.

Ingestion is an administrative operation: it changes what every future answer
is grounded in. Uploads are validated on extension, MIME type and size before
a single byte is parsed.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, File, Form, UploadFile, status

from ..core.errors import ERROR_RESPONSES, DocumentError, DocumentNotFoundError
from ..core.logging import get_logger
from ..rag.ingestion import IngestionPipeline
from ..rag.loaders import SUPPORTED_MIME_TYPES, SUPPORTED_SUFFIXES
from ..schemas.rag import (
    DocumentDeleteResponse,
    DocumentListResponse,
    IngestedDocument,
    IngestResponse,
)
from .deps import EmbedderDep, SettingsDep, VectorStoreDep

logger = get_logger(__name__)

#: Starlette renamed 413 in newer releases; support both spellings.
HTTP_413_TOO_LARGE: int = getattr(
    status, "HTTP_413_CONTENT_TOO_LARGE", 413
)

router = APIRouter(prefix="/documents", tags=["Documents"], responses=ERROR_RESPONSES)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List indexed documents",
    description="Every document currently present in the knowledge collection, "
    "with its chunk count.",
)
def list_documents(
    settings: SettingsDep, store: VectorStoreDep
) -> DocumentListResponse:
    stats = store.stats()
    documents = store.list_documents() if stats.exists else []
    return DocumentListResponse(
        collection=stats.collection,
        document_count=len(documents),
        vector_count=stats.vector_count,
        documents=documents,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest an uploaded document",
    description=(
        "Uploads one `.md`, `.txt` or `.pdf` file, chunks it, embeds it and "
        "stores it in the knowledge collection. Re-ingesting the same document "
        "replaces its existing chunks rather than duplicating them."
    ),
)
async def ingest_upload(
    settings: SettingsDep,
    store: VectorStoreDep,
    embedder: EmbedderDep,
    file: UploadFile = File(description="Document to ingest (.md, .txt or .pdf)."),
    category: str | None = Form(default=None, description="Knowledge area."),
    title: str | None = Form(default=None, description="Override the title."),
) -> IngestResponse:
    filename = file.filename or "upload"
    _validate_upload(filename, file.content_type)

    data = await file.read()
    if not data:
        raise DocumentError(f"'{filename}' is empty.", code="EMPTY_DOCUMENT")
    if len(data) > settings.max_upload_bytes:
        raise DocumentError(
            f"'{filename}' exceeds the {settings.max_upload_bytes} byte limit.",
            code="DOCUMENT_TOO_LARGE",
            status_code=HTTP_413_TOO_LARGE,
            details={"max_bytes": settings.max_upload_bytes},
        )

    pipeline = IngestionPipeline(embedder, store, settings)
    result = pipeline.ingest_bytes(data, filename, category=category, title=title)

    return IngestResponse(
        documents=[IngestedDocument(**asdict(result))],
        total_chunks=result.chunks_written,
        duration_ms=result.duration_ms,
    )


@router.post(
    "/reindex",
    response_model=IngestResponse,
    summary="Re-ingest the bundled knowledge base",
    description=(
        "Re-runs ingestion over `KNOWLEDGE_DIR`. Equivalent to "
        "`python -m app.rag.ingest`, exposed for administrators who cannot "
        "reach a shell."
    ),
)
def reindex(
    settings: SettingsDep, store: VectorStoreDep, embedder: EmbedderDep
) -> IngestResponse:
    pipeline = IngestionPipeline(embedder, store, settings)
    report = pipeline.ingest_directory(settings.knowledge_path)

    return IngestResponse(
        documents=[IngestedDocument(**asdict(item)) for item in report.documents],
        failures=[
            {"source": source, "message": message}
            for source, message in report.failures
        ],
        total_chunks=report.chunk_count,
        duration_ms=report.duration_ms,
    )


@router.delete(
    "/{document_id:path}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Removes every chunk belonging to the document from the "
    "knowledge collection.",
)
def delete_document(
    document_id: str, settings: SettingsDep, store: VectorStoreDep
) -> DocumentDeleteResponse:
    deleted = store.delete_document(document_id)
    if deleted == 0:
        raise DocumentNotFoundError(
            f"No document '{document_id}' is indexed.",
            details={"document_id": document_id},
        )

    return DocumentDeleteResponse(document_id=document_id, chunks_deleted=deleted)


def _validate_upload(filename: str, content_type: str | None) -> None:
    """Reject unsupported uploads before reading the body."""
    from pathlib import Path

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            f"Unsupported file type '{suffix or filename}'.",
            code="UNSUPPORTED_FILE_TYPE",
            details={"supported": sorted(SUPPORTED_SUFFIXES)},
        )

    # The extension is authoritative; the MIME check catches obvious mismatches
    # without rejecting browsers that report .md as application/octet-stream.
    if content_type:
        base_type = content_type.split(";")[0].strip().lower()
        if base_type not in SUPPORTED_MIME_TYPES:
            raise DocumentError(
                f"Unsupported content type '{base_type}'.",
                code="UNSUPPORTED_MEDIA_TYPE",
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                details={"supported": sorted(SUPPORTED_MIME_TYPES)},
            )
