"""Documents API tests: upload validation, listing and deletion."""

import pytest

from app.rag.embeddings import get_embedder
from app.rag.vector_store import VectorStore, get_vector_store
from tests.test_ingestion import StubEmbedder
from tests.test_vector_store import StubClient, StubPoint, make_chunk


@pytest.fixture(name="doc_client")
def doc_client_fixture(client, settings):
    """Client wired to a stub vector store and a stub embedder."""
    stub_client = StubClient(existing=True)
    store = VectorStore(settings=settings, client=stub_client)
    client.app.dependency_overrides[get_vector_store] = lambda: store
    client.app.dependency_overrides[get_embedder] = lambda: StubEmbedder()
    return client, stub_client


MARKDOWN = (
    b"---\ntitle: Uploaded Guide\ncategory: lab_tests\n---\n"
    b"# Uploaded Guide\n\nBody text about platelets and plasma."
)


# ------------------------------------------------------------------ upload --
def test_ingest_markdown_upload(doc_client):
    client, _ = doc_client

    response = client.post(
        "/documents/ingest",
        files={"file": ("guide.md", MARKDOWN, "text/markdown")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["total_chunks"] >= 1
    assert body["documents"][0]["title"] == "Uploaded Guide"
    assert body["documents"][0]["source"] == "guide.md"


def test_ingest_plain_text_upload(doc_client):
    client, _ = doc_client

    response = client.post(
        "/documents/ingest",
        files={
            "file": (
                "notes.txt",
                b"Plain text about sample handling.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201


def test_ingest_accepts_octet_stream_for_markdown(doc_client):
    """Browsers commonly send .md as application/octet-stream."""
    client, _ = doc_client

    response = client.post(
        "/documents/ingest",
        files={"file": ("guide.md", MARKDOWN, "application/octet-stream")},
    )

    assert response.status_code == 201


def test_ingest_rejects_unsupported_extension(doc_client):
    client, _ = doc_client

    response = client.post(
        "/documents/ingest",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_ingest_rejects_unsupported_mime_type(doc_client):
    client, _ = doc_client

    response = client.post(
        "/documents/ingest",
        files={"file": ("guide.md", MARKDOWN, "image/png")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_ingest_rejects_empty_file(doc_client):
    client, _ = doc_client

    response = client.post(
        "/documents/ingest", files={"file": ("blank.md", b"", "text/markdown")}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_DOCUMENT"


def test_ingest_enforces_the_size_limit(doc_client, settings):
    client, _ = doc_client
    settings.max_upload_bytes = 64

    response = client.post(
        "/documents/ingest",
        files={"file": ("big.md", b"# H\n\n" + b"x " * 200, "text/markdown")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "DOCUMENT_TOO_LARGE"


def test_ingest_honours_explicit_category(doc_client):
    client, _ = doc_client

    response = client.post(
        "/documents/ingest",
        files={"file": ("guide.md", MARKDOWN, "text/markdown")},
        data={"category": "faq"},
    )

    assert response.json()["documents"][0]["category"] == "faq"


def test_ingest_missing_file_is_a_validation_error(doc_client):
    client, _ = doc_client

    response = client.post("/documents/ingest")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


# -------------------------------------------------------------------- list --
def test_list_documents_returns_indexed_documents(doc_client):
    client, stub = doc_client
    stub.count_value = 2
    stub.points = [
        StubPoint(make_chunk(0).to_payload()),
        StubPoint(make_chunk(1).to_payload()),
    ]

    body = client.get("/documents").json()

    assert body["collection"] == "test_knowledge"
    assert body["document_count"] == 1
    assert body["documents"][0]["chunk_count"] == 2


def test_list_documents_on_empty_collection(client, settings):
    store = VectorStore(settings=settings, client=StubClient(existing=False))
    client.app.dependency_overrides[get_vector_store] = lambda: store

    body = client.get("/documents").json()

    assert body["document_count"] == 0
    assert body["documents"] == []


# ------------------------------------------------------------------ delete --
def test_delete_document_removes_chunks(doc_client):
    client, stub = doc_client
    stub.count_value = 3

    response = client.delete("/documents/lab_tests/cbc")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "lab_tests/cbc",
        "chunks_deleted": 3,
    }


def test_delete_unknown_document_returns_404(doc_client):
    client, stub = doc_client
    stub.count_value = 0

    response = client.delete("/documents/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


# ----------------------------------------------------------------- reindex --
def test_reindex_ingests_the_bundled_knowledge_base(doc_client):
    client, _ = doc_client

    response = client.post("/documents/reindex")

    assert response.status_code == 200
    body = response.json()
    assert len(body["documents"]) == 12
    assert body["total_chunks"] > 40
    assert body["failures"] == []
