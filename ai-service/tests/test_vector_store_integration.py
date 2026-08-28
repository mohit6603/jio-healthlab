"""Integration tests against a live Qdrant.

Skipped automatically when no server is reachable, so the default unit run
stays hermetic::

    QDRANT_TEST_URL=http://localhost:6333 pytest -m integration
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.config import Settings
from app.rag.vector_store import VectorStore
from app.schemas.rag import EmbeddedChunk, KnowledgeChunk

pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
DIMENSION = 8


def _server_available() -> bool:
    import httpx

    try:
        return httpx.get(f"{QDRANT_URL}/readyz", timeout=2.0).status_code < 500
    except Exception:  # noqa: BLE001 - any failure means "no server"
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _server_available(), reason=f"no Qdrant at {QDRANT_URL}"),
]


@pytest.fixture(name="store")
def store_fixture():
    """A store bound to a throwaway collection, dropped afterwards."""
    collection = f"itest_{uuid.uuid4().hex[:10]}"
    settings = Settings(
        qdrant_url=QDRANT_URL,
        qdrant_collection=collection,
        qdrant_report_collection=f"{collection}_reports",
        embedding_dimension=DIMENSION,
        log_json=False,
    )
    store = VectorStore(settings=settings)
    yield store
    for name in (collection, f"{collection}_reports"):
        try:
            store.drop_collection(name)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    store.close()


def _chunk(index: int, document_id: str = "cbc", text: str = "text") -> KnowledgeChunk:
    return KnowledgeChunk(
        document_id=document_id,
        chunk_id=f"{document_id}::{index}",
        chunk_index=index,
        title=f"{document_id.upper()} Guide",
        source=f"{document_id}.md",
        category="lab_tests",
        section="Overview",
        text=text,
    )


def _vec(*values: float) -> list[float]:
    padded = list(values) + [0.0] * (DIMENSION - len(values))
    return padded[:DIMENSION]


def test_collection_is_created_on_demand(store):
    name = store.ensure_collection()

    assert store.client.collection_exists(name)


def test_upsert_then_search_round_trip(store):
    store.upsert_chunks(
        [
            EmbeddedChunk(chunk=_chunk(0, text="red cells"), vector=_vec(1, 0)),
            EmbeddedChunk(chunk=_chunk(1, text="platelets"), vector=_vec(0, 1)),
        ]
    )

    hits = store.search(_vec(1, 0), top_k=1)

    assert len(hits) == 1
    assert hits[0].text == "red cells"
    assert hits[0].chunk_id == "cbc::0"
    assert hits[0].score > 0.9


def test_reingest_replaces_rather_than_duplicates(store):
    chunk = _chunk(0, text="first version")
    store.upsert_chunks([EmbeddedChunk(chunk=chunk, vector=_vec(1, 0))])
    store.upsert_chunks(
        [
            EmbeddedChunk(
                chunk=_chunk(0, text="second version"), vector=_vec(1, 0)
            )
        ]
    )

    stats = store.stats()
    hits = store.search(_vec(1, 0), top_k=5)

    assert stats.vector_count == 1
    assert hits[0].text == "second version"


def test_score_threshold_filters_weak_matches(store):
    store.upsert_chunks(
        [EmbeddedChunk(chunk=_chunk(0, text="red cells"), vector=_vec(1, 0))]
    )

    assert store.search(_vec(0, 1), top_k=5, score_threshold=0.5) == []


def test_metadata_filter_restricts_results(store):
    store.upsert_chunks(
        [
            EmbeddedChunk(chunk=_chunk(0, "cbc", "cbc text"), vector=_vec(1, 0)),
            EmbeddedChunk(chunk=_chunk(0, "tsh", "tsh text"), vector=_vec(1, 0)),
        ]
    )

    hits = store.search(_vec(1, 0), top_k=5, filters={"document_id": "tsh"})

    assert [hit.document_id for hit in hits] == ["tsh"]


def test_delete_document_removes_only_that_document(store):
    store.upsert_chunks(
        [
            EmbeddedChunk(chunk=_chunk(0, "cbc"), vector=_vec(1, 0)),
            EmbeddedChunk(chunk=_chunk(1, "cbc"), vector=_vec(0, 1)),
            EmbeddedChunk(chunk=_chunk(0, "tsh"), vector=_vec(0, 0, 1)),
        ]
    )

    removed = store.delete_document("cbc")

    assert removed == 2
    assert store.stats().vector_count == 1
    assert {doc.document_id for doc in store.list_documents()} == {"tsh"}


def test_delete_unknown_document_is_a_no_op(store):
    store.ensure_collection()

    assert store.delete_document("nope") == 0


def test_list_documents_groups_and_counts(store):
    store.upsert_chunks(
        [
            EmbeddedChunk(chunk=_chunk(0, "cbc"), vector=_vec(1, 0)),
            EmbeddedChunk(chunk=_chunk(1, "cbc"), vector=_vec(0, 1)),
            EmbeddedChunk(chunk=_chunk(0, "tsh"), vector=_vec(0, 0, 1)),
        ]
    )

    documents = {doc.document_id: doc.chunk_count for doc in store.list_documents()}

    assert documents == {"cbc": 2, "tsh": 1}


def test_knowledge_and_report_collections_do_not_mix(store):
    store.upsert_chunks(
        [EmbeddedChunk(chunk=_chunk(0, "cbc"), vector=_vec(1, 0))]
    )
    store.upsert_chunks(
        [EmbeddedChunk(chunk=_chunk(0, "rep"), vector=_vec(1, 0))],
        collection=store.report_collection,
    )

    knowledge = store.search(_vec(1, 0), top_k=5)
    reports = store.search(_vec(1, 0), top_k=5, collection=store.report_collection)

    assert [hit.document_id for hit in knowledge] == ["cbc"]
    assert [hit.document_id for hit in reports] == ["rep"]


def test_health_reports_live_collection(store):
    store.upsert_chunks(
        [EmbeddedChunk(chunk=_chunk(0, "cbc"), vector=_vec(1, 0))]
    )

    report = store.health()

    assert report.reachable is True
    assert report.collection_exists is True
    assert report.vector_count == 1


def test_health_on_unreachable_server_does_not_raise():
    settings = Settings(qdrant_url="http://127.0.0.1:59999", log_json=False)
    offline = VectorStore(settings=settings)

    report = offline.health()

    assert report.reachable is False
    assert report.detail
