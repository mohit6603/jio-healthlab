"""Vector-store unit tests.

These run against a stub client so they need no Qdrant. Behaviour against a
live server is covered by ``test_vector_store_integration.py``.
"""

from typing import Any

import pytest
from qdrant_client.http import models as qmodels

from app.core.errors import VectorStoreError
from app.rag.vector_store import (
    POINT_NAMESPACE,
    VectorStore,
    point_id,
)
from app.schemas.rag import EmbeddedChunk, KnowledgeChunk


class StubPoint:
    def __init__(self, payload: dict[str, Any], score: float = 0.9, id_: str = "p1"):
        self.payload = payload
        self.score = score
        self.id = id_


class StubResponse:
    def __init__(self, points: list[StubPoint]):
        self.points = points


class StubCount:
    def __init__(self, count: int):
        self.count = count


class StubClient:
    """Records calls so tests can assert on what was sent to Qdrant."""

    def __init__(self, *, existing: bool = False, fail_on: str | None = None):
        self.existing = existing
        self.fail_on = fail_on
        self.created: list[str] = []
        self.upserted: list[Any] = []
        self.deleted: list[Any] = []
        self.indexes: list[tuple[str, str]] = []
        self.queries: list[dict[str, Any]] = []
        self.points: list[StubPoint] = []
        self.count_value = 0
        self.closed = False

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            raise RuntimeError(f"boom in {name}")

    def collection_exists(self, name: str) -> bool:
        self._maybe_fail("collection_exists")
        return self.existing

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self._maybe_fail("create_collection")
        self.created.append(collection_name)
        self.vectors_config = vectors_config
        self.existing = True

    def create_payload_index(self, collection_name: str, field_name: str, **_: Any) -> None:
        self.indexes.append((collection_name, field_name))

    def upsert(self, collection_name: str, points: list[Any], wait: bool = True) -> None:
        self._maybe_fail("upsert")
        self.upserted.extend(points)

    def query_points(self, **kwargs: Any) -> StubResponse:
        self._maybe_fail("query_points")
        self.queries.append(kwargs)
        return StubResponse(self.points)

    def count(self, collection_name: str, count_filter: Any = None, exact: bool = True):
        self._maybe_fail("count")
        return StubCount(self.count_value)

    def delete(self, collection_name: str, points_selector: Any, wait: bool = True) -> None:
        self._maybe_fail("delete")
        self.deleted.append(points_selector)

    def scroll(self, collection_name: str, limit: int, offset: Any = None, **_: Any):
        self._maybe_fail("scroll")
        return list(self.points), None

    def get_collections(self):
        self._maybe_fail("get_collections")
        return object()

    def delete_collection(self, collection_name: str) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def make_chunk(index: int = 0, document_id: str = "cbc") -> KnowledgeChunk:
    return KnowledgeChunk(
        document_id=document_id,
        chunk_id=f"{document_id}::{index}",
        chunk_index=index,
        title="Complete Blood Count Guide",
        source="cbc.md",
        category="lab_tests",
        section="What it measures",
        text="A CBC measures red blood cells, white blood cells and platelets.",
    )


@pytest.fixture(name="store_and_client")
def store_and_client_fixture(settings):
    client = StubClient()
    return VectorStore(settings=settings, client=client), client


# ------------------------------------------------------------- ids ---------
def test_point_id_is_deterministic():
    assert point_id("cbc::0") == point_id("cbc::0")


def test_point_id_differs_per_chunk():
    assert point_id("cbc::0") != point_id("cbc::1")


def test_point_id_uses_the_declared_namespace():
    import uuid

    assert point_id("cbc::0") == str(uuid.uuid5(POINT_NAMESPACE, "cbc::0"))


# ------------------------------------------------------ collections --------
def test_ensure_collection_creates_when_missing(store_and_client):
    store, client = store_and_client

    name = store.ensure_collection()

    assert name == "test_knowledge"
    assert client.created == ["test_knowledge"]
    assert client.vectors_config.size == 384
    assert client.vectors_config.distance == qmodels.Distance.COSINE


def test_ensure_collection_creates_payload_indexes(store_and_client):
    store, client = store_and_client

    store.ensure_collection()

    fields = {field for _, field in client.indexes}
    assert fields == {"document_id", "category", "source"}


def test_ensure_collection_is_memoised(store_and_client):
    store, client = store_and_client

    store.ensure_collection()
    store.ensure_collection()

    assert client.created == ["test_knowledge"]


def test_ensure_collection_skips_creation_when_present(settings):
    client = StubClient(existing=True)
    store = VectorStore(settings=settings, client=client)

    store.ensure_collection()

    assert client.created == []


def test_ensure_collection_wraps_driver_failure(settings):
    store = VectorStore(settings=settings, client=StubClient(fail_on="collection_exists"))

    with pytest.raises(VectorStoreError):
        store.ensure_collection()


def test_knowledge_and_report_collections_are_separate(store_and_client):
    store, _ = store_and_client

    assert store.knowledge_collection == "test_knowledge"
    assert store.report_collection == "test_reports"
    assert store.knowledge_collection != store.report_collection


# ----------------------------------------------------------- upsert --------
def test_upsert_writes_points_with_full_payload(store_and_client):
    store, client = store_and_client
    chunk = make_chunk()

    written = store.upsert_chunks([EmbeddedChunk(chunk=chunk, vector=[0.1] * 384)])

    assert written == 1
    point = client.upserted[0]
    assert point.id == point_id("cbc::0")
    assert point.payload["document_id"] == "cbc"
    assert point.payload["title"] == "Complete Blood Count Guide"
    assert point.payload["source"] == "cbc.md"
    assert point.payload["category"] == "lab_tests"
    assert point.payload["section"] == "What it measures"
    assert "red blood cells" in point.payload["text"]


def test_upsert_of_empty_sequence_is_a_no_op(store_and_client):
    store, client = store_and_client

    assert store.upsert_chunks([]) == 0
    assert client.upserted == []


def test_reingesting_the_same_chunk_reuses_its_id(store_and_client):
    store, client = store_and_client
    chunk = make_chunk()

    store.upsert_chunks([EmbeddedChunk(chunk=chunk, vector=[0.1] * 384)])
    store.upsert_chunks([EmbeddedChunk(chunk=chunk, vector=[0.2] * 384)])

    assert client.upserted[0].id == client.upserted[1].id


def test_upsert_wraps_driver_failure(settings):
    store = VectorStore(settings=settings, client=StubClient(fail_on="upsert"))

    with pytest.raises(VectorStoreError):
        store.upsert_chunks([EmbeddedChunk(chunk=make_chunk(), vector=[0.1] * 384)])


# ----------------------------------------------------------- search --------
def test_search_maps_payload_into_hits(store_and_client):
    store, client = store_and_client
    client.points = [StubPoint(make_chunk().to_payload(), score=0.87)]

    hits = store.search([0.1] * 384)

    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.87)
    assert hits[0].source == "cbc.md"
    assert hits[0].title == "Complete Blood Count Guide"
    assert hits[0].chunk_id == "cbc::0"


def test_search_defaults_to_configured_top_k(store_and_client):
    store, client = store_and_client

    store.search([0.1] * 384)

    assert client.queries[0]["limit"] == 5


def test_search_honours_explicit_top_k_and_threshold(store_and_client):
    store, client = store_and_client

    store.search([0.1] * 384, top_k=3, score_threshold=0.4)

    assert client.queries[0]["limit"] == 3
    assert client.queries[0]["score_threshold"] == 0.4


def test_search_builds_equality_filter(store_and_client):
    store, client = store_and_client

    store.search([0.1] * 384, filters={"category": "lab_tests"})

    condition = client.queries[0]["query_filter"].must[0]
    assert condition.key == "category"
    assert condition.match.value == "lab_tests"


def test_search_builds_any_filter_for_lists(store_and_client):
    store, client = store_and_client

    store.search([0.1] * 384, filters={"category": ["lab_tests", "faq"]})

    condition = client.queries[0]["query_filter"].must[0]
    assert condition.match.any == ["lab_tests", "faq"]


def test_search_without_filters_sends_none(store_and_client):
    store, client = store_and_client

    store.search([0.1] * 384)

    assert client.queries[0]["query_filter"] is None


def test_search_ignores_none_valued_filters(store_and_client):
    store, client = store_and_client

    store.search([0.1] * 384, filters={"category": None})

    assert client.queries[0]["query_filter"] is None


def test_search_targets_the_requested_collection(store_and_client):
    store, client = store_and_client

    store.search([0.1] * 384, collection="test_reports")

    assert client.queries[0]["collection_name"] == "test_reports"


def test_search_wraps_driver_failure(settings):
    store = VectorStore(settings=settings, client=StubClient(fail_on="query_points"))

    with pytest.raises(VectorStoreError):
        store.search([0.1] * 384)


# ----------------------------------------------------------- delete --------
def test_delete_document_filters_by_document_id(store_and_client):
    store, client = store_and_client
    client.existing = True
    client.count_value = 4

    removed = store.delete_document("cbc")

    assert removed == 4
    condition = client.deleted[0].filter.must[0]
    assert condition.key == "document_id"
    assert condition.match.value == "cbc"


def test_delete_document_skips_when_nothing_indexed(store_and_client):
    store, client = store_and_client
    client.existing = True
    client.count_value = 0

    assert store.delete_document("missing") == 0
    assert client.deleted == []


def test_delete_document_wraps_driver_failure(settings):
    client = StubClient(existing=True, fail_on="count")
    store = VectorStore(settings=settings, client=client)

    with pytest.raises(VectorStoreError):
        store.delete_document("cbc")


def test_delete_document_on_missing_collection_returns_zero(store_and_client):
    """Ingesting into a fresh deployment deletes before the collection exists."""
    store, client = store_and_client
    client.existing = False

    assert store.delete_document("cbc") == 0
    assert client.deleted == []


# ------------------------------------------------------------ browse -------
def test_list_documents_groups_chunks(store_and_client):
    store, client = store_and_client
    client.points = [
        StubPoint(make_chunk(0).to_payload()),
        StubPoint(make_chunk(1).to_payload()),
        StubPoint(make_chunk(0, document_id="thyroid").to_payload()),
    ]

    documents = store.list_documents()

    counts = {doc.document_id: doc.chunk_count for doc in documents}
    assert counts == {"cbc": 2, "thyroid": 1}


def test_stats_reports_missing_collection(settings):
    client = StubClient(existing=False)
    store = VectorStore(settings=settings, client=client)

    stats = store.stats()

    assert stats.exists is False
    assert stats.vector_count == 0


def test_stats_counts_vectors_and_documents(settings):
    client = StubClient(existing=True)
    client.count_value = 2
    client.points = [
        StubPoint(make_chunk(0).to_payload()),
        StubPoint(make_chunk(0, document_id="thyroid").to_payload()),
    ]
    store = VectorStore(settings=settings, client=client)

    stats = store.stats()

    assert stats.vector_count == 2
    assert stats.document_count == 2
    assert stats.dimension == 384


# ------------------------------------------------------------ health -------
def test_health_reports_reachable(settings):
    client = StubClient(existing=True)
    client.count_value = 7
    store = VectorStore(settings=settings, client=client)

    report = store.health()

    assert report.reachable is True
    assert report.collection_exists is True
    assert report.vector_count == 7


def test_health_reports_unreachable_without_raising(settings):
    store = VectorStore(settings=settings, client=StubClient(fail_on="get_collections"))

    report = store.health()

    assert report.reachable is False
    assert "boom" in report.detail
