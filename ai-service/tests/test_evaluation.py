"""Tests for the evaluation harness.

The harness produces the numbers this project reports, so its scoring has to
be right. These tests check the metric arithmetic against hand-computed
values -- they do not run a model.
"""

from __future__ import annotations

import json

import pytest

from evaluation.dataset import (
    BOUNDARY_KINDS,
    NO_RETRIEVAL_KINDS,
    load_questions,
)
from evaluation.evaluate_rag import RagResult
from evaluation.evaluate_rag import summarise as summarise_rag
from evaluation.evaluate_retrieval import QuestionResult
from evaluation.evaluate_retrieval import summarise as summarise_retrieval


# --------------------------------------------------------------- dataset ---
def test_question_set_loads():
    questions = load_questions()

    assert len(questions) >= 40


def test_every_question_has_a_unique_id():
    ids = [item.id for item in load_questions()]

    assert len(ids) == len(set(ids))


def test_factual_questions_name_a_source_and_keywords():
    for item in load_questions():
        if item.kind != "factual":
            continue
        assert item.expected_source, item.id
        assert item.expected_keywords, item.id


def test_negative_questions_expect_no_source():
    for item in load_questions():
        if item.kind == "factual":
            continue
        assert item.expected_source is None, item.id


def test_expected_sources_exist_in_the_knowledge_base(settings):
    """An expectation pointing at a file that does not exist is untestable."""
    available = {
        path.name for path in settings.knowledge_path.rglob("*") if path.is_file()
    }

    for item in load_questions():
        if item.expected_source:
            assert item.expected_source in available, item.id


def test_expected_keywords_appear_in_their_source_document(settings):
    """A keyword absent from the cited document could never be produced from it."""
    missing = []
    for item in load_questions():
        if not item.expected_source:
            continue
        matches = list(settings.knowledge_path.rglob(item.expected_source))
        assert matches, item.expected_source
        text = matches[0].read_text(encoding="utf-8").lower()
        for keyword in item.expected_keywords:
            if keyword.lower() not in text:
                missing.append(f"{item.id}: {keyword!r}")

    assert not missing, "keywords absent from their source document: " + "; ".join(
        missing
    )


def test_kind_partitions_are_disjoint():
    assert NO_RETRIEVAL_KINDS.isdisjoint(BOUNDARY_KINDS)


# ------------------------------------------------------ retrieval scoring --
def result(
    ident: str,
    kind: str = "factual",
    expected: str | None = "cbc.md",
    rank: int | None = 1,
    hits: int = 5,
) -> QuestionResult:
    item = QuestionResult(
        id=ident, kind=kind, question="q", expected_source=expected
    )
    item.rank = rank
    item.retrieval_count = hits
    item.total_ms = 5.0
    return item


def test_hit_at_k_counts_ranks_within_the_cutoff():
    results = [
        result("a", rank=1),
        result("b", rank=3),
        result("c", rank=None),
        result("d", rank=2),
    ]

    summary = summarise_retrieval(results, top_k=5)

    assert summary["hit_at_1"] == 0.25
    assert summary["hit_at_3"] == 0.75
    assert summary["hit_at_5"] == 0.75


def test_mrr_counts_a_miss_as_zero():
    """MRR must not be conditioned on success, or it overstates quality."""
    results = [result("a", rank=1), result("b", rank=2), result("c", rank=None)]

    summary = summarise_retrieval(results, top_k=5)

    assert summary["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3, abs=1e-4)


def test_perfect_retrieval_scores_one():
    results = [result(str(index), rank=1) for index in range(5)]

    summary = summarise_retrieval(results, top_k=5)

    assert summary["hit_at_1"] == 1.0
    assert summary["mrr"] == 1.0


def test_rejection_rate_uses_unanswerable_questions_only():
    results = [
        result("a", rank=1),
        result("oos", kind="out_of_domain", expected=None, rank=None, hits=0),
        result("inj", kind="prompt_injection", expected=None, rank=None, hits=0),
        # A boundary question retrieving content is correct, not a failure.
        result("bnd", kind="clinical_boundary", expected=None, rank=None, hits=5),
    ]

    summary = summarise_retrieval(results, top_k=5)

    assert summary["rejection_rate"] == 1.0
    assert summary["false_retrievals"] == 0
    assert summary["boundary_retrieved"] == 1


def test_a_retrieved_unanswerable_question_lowers_the_rejection_rate():
    results = [
        result("oos1", kind="out_of_domain", expected=None, rank=None, hits=0),
        result("oos2", kind="out_of_domain", expected=None, rank=None, hits=3),
    ]

    summary = summarise_retrieval(results, top_k=5)

    assert summary["rejection_rate"] == 0.5
    assert summary["false_retrievals"] == 1


def test_latency_statistics_are_reported():
    results = [result("a"), result("b")]
    results[0].total_ms = 10.0
    results[1].total_ms = 20.0

    summary = summarise_retrieval(results, top_k=5)

    assert summary["latency_ms"]["mean"] == 15.0
    assert summary["latency_ms"]["max"] == 20.0


def test_empty_run_does_not_divide_by_zero():
    summary = summarise_retrieval([], top_k=5)

    assert summary["mrr"] == 0.0
    assert summary["rejection_rate"] == 0.0


# ------------------------------------------------------------ rag scoring --
def rag(
    ident: str,
    kind: str = "factual",
    answer: str = "A CBC measures red blood cells.",
    grounded: bool = True,
    sources: int = 3,
    keywords: list[str] | None = None,
    generation_ms: float = 100.0,
) -> RagResult:
    item = RagResult(id=ident, kind=kind, question="q", answer=answer)
    item.grounded = grounded
    item.source_count = sources
    item.expected_keywords = keywords or ["red blood cells"]
    lowered = answer.lower()
    item.matched_keywords = [k for k in item.expected_keywords if k.lower() in lowered]
    item.generation_ms = generation_ms
    return item


def test_keyword_coverage_is_the_matched_fraction():
    item = rag(
        "a",
        answer="It measures red blood cells and platelets.",
        keywords=["red blood cells", "platelets", "white blood cells"],
    )

    assert item.keyword_coverage == pytest.approx(2 / 3)


def test_source_presence_counts_answers_with_citations():
    results = [rag("a", sources=3), rag("b", sources=0), rag("c", sources=1)]

    summary = summarise_rag(results)

    assert summary["source_presence"] == pytest.approx(2 / 3, abs=1e-4)


def test_refusal_accuracy_requires_the_fixed_message():
    from app.rag.prompts import INSUFFICIENT_CONTEXT_MESSAGE

    good = rag(
        "oos1",
        kind="out_of_domain",
        answer=INSUFFICIENT_CONTEXT_MESSAGE,
        grounded=False,
        sources=0,
        keywords=[],
    )
    bad = rag(
        "oos2",
        kind="out_of_domain",
        answer="Brazil won in 1994.",
        grounded=True,
        sources=0,
        keywords=[],
    )

    summary = summarise_rag([good, bad])

    assert summary["refusal_accuracy"] == 0.5


def test_boundary_accuracy_requires_generation_to_be_skipped():
    from app.rag.safety import CLINICAL_BOUNDARY_MESSAGE

    good = rag(
        "b1",
        kind="clinical_boundary",
        answer=CLINICAL_BOUNDARY_MESSAGE,
        grounded=False,
        sources=0,
        keywords=[],
        generation_ms=0.0,
    )
    # Right message, but the model was still called -- not what the design says.
    bad = rag(
        "b2",
        kind="clinical_boundary",
        answer=CLINICAL_BOUNDARY_MESSAGE,
        grounded=False,
        sources=0,
        keywords=[],
        generation_ms=500.0,
    )

    summary = summarise_rag([good, bad])

    assert summary["boundary_accuracy"] == 0.5


def test_system_prompt_leak_is_detected():
    leaked = rag(
        "inj",
        kind="prompt_injection",
        answer="HANDLING THE REFERENCE MATERIAL - everything between ...",
        keywords=[],
    )

    summary = summarise_rag([leaked])

    assert summary["system_prompt_leaks"] == 1
    assert summary["injection_safe"] is False


def test_clean_answers_report_no_leak():
    summary = summarise_rag([rag("a"), rag("b")])

    assert summary["system_prompt_leaks"] == 0
    assert summary["injection_safe"] is True


def test_rag_latency_is_reported():
    summary = summarise_rag([rag("a", generation_ms=100.0), rag("b", generation_ms=300.0)])

    assert summary["latency_ms"]["generation_mean"] == 200.0
    assert summary["latency_ms"]["answers_generated"] == 2


def test_rag_summary_records_the_model(settings):
    summary = summarise_rag([rag("a")])

    assert summary["model"]
    assert summary["provider"]


def test_evaluation_results_serialise_to_json():
    """The harness writes JSON; the dataclasses must survive it."""
    from dataclasses import asdict

    payload = {
        "retrieval": [asdict(result("a"))],
        "rag": [asdict(rag("b"))],
    }

    assert json.loads(json.dumps(payload))


# ------------------------------------------------- spelling variants -------
def test_american_spelling_counts_as_a_match():
    """A miss caused by orthography measures spelling, not faithfulness."""
    from evaluation.dataset import keyword_matches

    assert keyword_matches("haemoglobin", "a decrease in hemoglobin levels")
    assert keyword_matches("anaemia", "the patient has anemia")


def test_exact_spelling_still_matches():
    from evaluation.dataset import keyword_matches

    assert keyword_matches("haemoglobin", "haemoglobin is low")


def test_an_unrelated_word_does_not_match():
    from evaluation.dataset import keyword_matches

    assert keyword_matches("haemoglobin", "platelets and white cells") is False


def test_variants_only_apply_to_listed_words():
    from evaluation.dataset import keyword_matches

    assert keyword_matches("platelets", "plateletes") is False
