"""Report index tests: collection separation, filtering, payload hygiene."""

from __future__ import annotations

import pytest

from app.rag.embeddings import get_embedder
from app.rag.report_index import (
    REPORT_CATEGORY,
    IndexedReport,
    ReportIndexer,
    report_document_id,
)
from app.rag.vector_store import VectorStore, get_vector_store
from app.schemas.rag import KnowledgeChunk
from tests.test_ingestion import StubEmbedder
from tests.test_vector_store import StubClient, StubPoint


@pytest.fixture(name="indexer_parts")
def indexer_parts_fixture(settings):
    stub = StubClient(existing=True)
    store = VectorStore(settings=settings, client=stub)
    return ReportIndexer(StubEmbedder(), store, settings), stub


def report_payload(report_id: int, text: str = "Urgent CBC at Andheri Hub.") -> dict:
    return KnowledgeChunk(
        document_id=report_document_id(report_id),
        chunk_id=report_document_id(report_id),
        chunk_index=0,
        title=f"Report #{report_id}",
        source="reports",
        category=REPORT_CATEGORY,
        text=text,
        extra={"status": "processing", "priority": "urgent"},
    ).to_payload()


# ------------------------------------------------------------- payload -----
def test_extras_are_flattened_into_the_payload():
    payload = report_payload(7)

    assert payload["status"] == "processing"
    assert payload["priority"] == "urgent"


def test_core_fields_cannot_be_shadowed_by_extras():
    chunk = KnowledgeChunk(
        document_id="real",
        chunk_id="real::0",
        chunk_index=0,
        title="Real",
        source="real.md",
        category="lab_tests",
        text="body",
        extra={"category": "spoofed", "text": "spoofed"},
    )

    payload = chunk.to_payload()

    assert payload["category"] == "lab_tests"
    assert payload["text"] == "body"


def test_document_id_is_stable():
    assert report_document_id(42) == "report:42"


# ------------------------------------------------------------- indexing ----
def test_indexing_writes_to_the_report_collection(indexer_parts, settings):
    indexer, stub = indexer_parts

    indexer.index([IndexedReport(report_id=1, text="Urgent CBC at Andheri Hub.")])

    assert indexer.collection == settings.qdrant_report_collection
    assert indexer.collection != settings.qdrant_collection
    # Assert on where the write actually went, not on collection creation --
    # the collection may already exist.
    assert stub.upsert_collections == [settings.qdrant_report_collection]


def test_indexed_points_are_tagged_as_reports(indexer_parts):
    indexer, stub = indexer_parts

    indexer.index([IndexedReport(report_id=1, text="Urgent CBC.")])

    assert stub.upserted[0].payload["category"] == REPORT_CATEGORY


def test_only_whitelisted_metadata_is_stored(indexer_parts):
    indexer, stub = indexer_parts

    indexer.index(
        [
            IndexedReport(
                report_id=1,
                text="Urgent CBC.",
                metadata={
                    "status": "processing",
                    "priority": "urgent",
                    "patient_name": "Asha Nair",
                    "phone": "+91 98765 10001",
                },
            )
        ]
    )

    payload = stub.upserted[0].payload
    assert payload["status"] == "processing"
    assert "patient_name" not in payload
    assert "phone" not in payload


def test_report_indexes_are_created(indexer_parts):
    indexer, stub = indexer_parts

    indexer.index([IndexedReport(report_id=1, text="Urgent CBC.")])

    fields = {field for _, field in stub.indexes}
    assert {"status", "priority", "branch", "city", "test_type"} <= fields


def test_reindexing_replaces_rather_than_duplicates(indexer_parts):
    indexer, stub = indexer_parts

    indexer.index([IndexedReport(report_id=1, text="first")])
    indexer.index([IndexedReport(report_id=1, text="second")])

    assert stub.upserted[0].id == stub.upserted[1].id


def test_empty_index_call_is_a_no_op(indexer_parts):
    indexer, stub = indexer_parts

    assert indexer.index([]) == 0
    assert stub.upserted == []


# --------------------------------------------------------------- search ----
def test_search_always_filters_to_the_report_category(indexer_parts):
    """Belt and braces: a knowledge chunk must never surface as a report."""
    indexer, stub = indexer_parts
    stub.points = []

    indexer.search("urgent kidney tests in Mumbai")

    conditions = {c.key: c.match.value for c in stub.queries[0]["query_filter"].must}
    assert conditions["category"] == REPORT_CATEGORY


def test_search_targets_the_report_collection(indexer_parts, settings):
    indexer, stub = indexer_parts
    stub.points = []

    indexer.search("query")

    assert stub.queries[0]["collection_name"] == settings.qdrant_report_collection


def test_search_applies_whitelisted_filters(indexer_parts):
    indexer, stub = indexer_parts
    stub.points = []

    indexer.search("query", filters={"priority": "urgent", "city": "Mumbai"})

    conditions = {c.key: c.match.value for c in stub.queries[0]["query_filter"].must}
    assert conditions["priority"] == "urgent"
    assert conditions["city"] == "Mumbai"


def test_search_ignores_unknown_filters(indexer_parts):
    indexer, stub = indexer_parts
    stub.points = []

    indexer.search("query", filters={"patient_name": "Asha Nair"})

    conditions = {c.key for c in stub.queries[0]["query_filter"].must}
    assert conditions == {"category"}


def test_search_maps_hits_to_report_ids(indexer_parts):
    indexer, stub = indexer_parts
    stub.points = [StubPoint(report_payload(42), score=0.77)]

    results = indexer.search("urgent cbc")

    assert results[0].report_id == 42
    assert results[0].score == pytest.approx(0.77)
    assert results[0].text == "Urgent CBC at Andheri Hub."


def test_hit_with_a_malformed_id_is_dropped(indexer_parts):
    indexer, stub = indexer_parts
    payload = report_payload(1)
    payload["document_id"] = "not-a-report"
    stub.points = [StubPoint(payload, score=0.9)]

    assert indexer.search("query") == []


def test_blank_query_is_rejected(indexer_parts):
    from app.rag.retriever import EmptyQueryError

    indexer, _ = indexer_parts

    with pytest.raises(EmptyQueryError):
        indexer.search("   ")


# --------------------------------------------------------------- delete ----
def test_delete_removes_the_report(indexer_parts):
    indexer, stub = indexer_parts
    stub.count_value = 1

    assert indexer.delete(42) == 1
    condition = stub.deleted[0].filter.must[0]
    assert condition.match.value == "report:42"


def test_deleting_an_unindexed_report_is_not_an_error(indexer_parts):
    indexer, stub = indexer_parts
    stub.count_value = 0

    assert indexer.delete(999) == 0


# ------------------------------------------------------------------ API ----
@pytest.fixture(name="report_client")
def report_client_fixture(client, settings):
    stub = StubClient(existing=True)
    store = VectorStore(settings=settings, client=stub)
    client.app.dependency_overrides[get_vector_store] = lambda: store
    client.app.dependency_overrides[get_embedder] = lambda: StubEmbedder()
    return client, stub


def test_index_endpoint(report_client):
    client, _ = report_client

    response = client.post(
        "/reports/index",
        json={"items": [{"report_id": 1, "text": "Urgent CBC at Andheri Hub."}]},
    )

    assert response.status_code == 201
    assert response.json()["indexed"] == 1


def test_search_endpoint(report_client):
    client, stub = report_client
    stub.points = [StubPoint(report_payload(42), score=0.77)]

    body = client.post(
        "/reports/search", json={"query": "urgent kidney tests in Mumbai"}
    ).json()

    assert body["retrieval_count"] == 1
    assert body["results"][0]["report_id"] == 42


def test_search_endpoint_rejects_a_blank_query(report_client):
    client, _ = report_client

    response = client.post("/reports/search", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_QUERY"


def test_index_endpoint_rejects_an_empty_batch(report_client):
    client, _ = report_client

    assert client.post("/reports/index", json={"items": []}).status_code == 422


def test_index_stats_endpoint(report_client):
    client, stub = report_client
    stub.count_value = 3
    stub.points = [StubPoint(report_payload(1)), StubPoint(report_payload(2))]

    body = client.get("/reports/index/stats").json()

    assert body["exists"] is True
    assert body["indexed_reports"] == 2
