"""Retrieval evaluation.

    python -m evaluation.evaluate_retrieval
    python -m evaluation.evaluate_retrieval --top-k 10 --json results.json

Measures retrieval alone -- no generation, so nothing here depends on the
language model. Metrics come from the run; none are hard-coded.

Reported
--------
Hit@K       fraction of factual questions where the expected source document
            appears in the top K results
MRR         mean reciprocal rank of the first correct source, over factual
            questions. A question that never retrieves its source contributes
            zero, so MRR is not conditioned on success
Rejection   fraction of negative questions that correctly retrieve nothing
            above the score threshold
Latency     embed, search and total, per question
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.errors import AIError
from app.core.logging import configure_logging
from app.rag.embeddings import get_embedder
from app.rag.retriever import Retriever
from app.rag.vector_store import get_vector_store

from .dataset import BOUNDARY_KINDS, NO_RETRIEVAL_KINDS, EvalQuestion, load_questions

#: Ranks reported for Hit@K.
HIT_CUTOFFS = (1, 3, 5)


@dataclass(slots=True)
class QuestionResult:
    """Outcome for one evaluated question."""

    id: str
    kind: str
    question: str
    expected_source: str | None
    retrieved_sources: list[str] = field(default_factory=list)
    top_score: float | None = None
    #: 1-based rank of the expected source, or None if it was never retrieved.
    rank: int | None = None
    retrieval_count: int = 0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


def evaluate_question(
    retriever: Retriever, question: EvalQuestion, top_k: int
) -> QuestionResult:
    """Retrieve for one question and score the result."""
    result = QuestionResult(
        id=question.id,
        kind=question.kind,
        question=question.question,
        expected_source=question.expected_source,
    )

    started = time.perf_counter()
    try:
        retrieval = retriever.retrieve(question.question, top_k=top_k)
    except AIError as exc:
        result.error = exc.code
        result.total_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    result.retrieved_sources = [hit.source for hit in retrieval.hits]
    result.retrieval_count = retrieval.count
    result.top_score = retrieval.best_score
    result.embed_ms = retrieval.embed_ms
    result.search_ms = retrieval.search_ms
    result.total_ms = retrieval.total_ms

    if question.expected_source:
        for index, source in enumerate(result.retrieved_sources, start=1):
            if source == question.expected_source:
                result.rank = index
                break

    return result


def summarise(results: list[QuestionResult], top_k: int) -> dict[str, Any]:
    """Aggregate per-question results. Every number is computed here."""
    factual = [item for item in results if item.expected_source is not None]
    # Only questions the knowledge base genuinely cannot answer. Clinical
    # boundary questions are excluded: they mention real topics, so retrieving
    # relevant chunks is correct, and refusing them happens later.
    no_retrieval = [item for item in results if item.kind in NO_RETRIEVAL_KINDS]
    boundary = [item for item in results if item.kind in BOUNDARY_KINDS]
    latencies = [item.total_ms for item in results if item.error is None]

    summary: dict[str, Any] = {
        "questions": len(results),
        "factual": len(factual),
        "no_retrieval_expected": len(no_retrieval),
        "boundary": len(boundary),
        "errors": sum(1 for item in results if item.error),
        "top_k": top_k,
        "score_threshold": get_settings().score_threshold,
    }

    for cutoff in HIT_CUTOFFS:
        if cutoff > top_k:
            continue
        hits = sum(1 for item in factual if item.hit_at(cutoff))
        summary[f"hit_at_{cutoff}"] = (
            round(hits / len(factual), 4) if factual else 0.0
        )

    summary["mrr"] = (
        round(sum(item.reciprocal_rank for item in factual) / len(factual), 4)
        if factual
        else 0.0
    )

    # Correct behaviour for an unanswerable question is to retrieve nothing.
    correct_rejections = sum(1 for item in no_retrieval if item.retrieval_count == 0)
    summary["rejection_rate"] = (
        round(correct_rejections / len(no_retrieval), 4) if no_retrieval else 0.0
    )
    summary["false_retrievals"] = len(no_retrieval) - correct_rejections
    # Reported, not scored: the safety layer handles these, and
    # evaluate_rag.py checks that it does.
    summary["boundary_retrieved"] = sum(
        1 for item in boundary if item.retrieval_count > 0
    )

    if latencies:
        ordered = sorted(latencies)
        summary["latency_ms"] = {
            "mean": round(statistics.fmean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 2),
            "max": round(max(latencies), 2),
        }
        summary["embed_ms_mean"] = round(
            statistics.fmean(item.embed_ms for item in results if item.error is None), 2
        )
        summary["search_ms_mean"] = round(
            statistics.fmean(
                item.search_ms for item in results if item.error is None
            ),
            2,
        )

    return summary


def print_report(results: list[QuestionResult], summary: dict[str, Any]) -> None:
    """Render the run to stdout."""
    print("\nPer-question retrieval")
    print(f"{'id':26} {'kind':18} {'rank':>5} {'score':>7} {'hits':>5} {'ms':>7}")
    print("-" * 74)
    for item in results:
        rank = str(item.rank) if item.rank else ("-" if item.expected_source else "n/a")
        score = f"{item.top_score:.3f}" if item.top_score is not None else "-"
        flag = ""
        if item.expected_source and item.rank is None:
            flag = "  <- MISS"
        elif item.kind in NO_RETRIEVAL_KINDS and item.retrieval_count:
            flag = "  <- should have retrieved nothing"
        elif item.kind in BOUNDARY_KINDS and item.retrieval_count:
            flag = "  (expected; refused by the safety layer)"
        print(
            f"{item.id:26} {item.kind:18} {rank:>5} {score:>7} "
            f"{item.retrieval_count:>5} {item.total_ms:>7.1f}{flag}"
        )

    print("\nSummary")
    print("-" * 74)
    print(
        f"  questions            {summary['questions']} "
        f"({summary['factual']} factual, "
        f"{summary['no_retrieval_expected']} unanswerable, "
        f"{summary['boundary']} clinical boundary)"
    )
    for cutoff in HIT_CUTOFFS:
        key = f"hit_at_{cutoff}"
        if key in summary:
            print(f"  Hit@{cutoff}                {summary[key]:.4f}")
    print(f"  MRR                  {summary['mrr']:.4f}")
    print(
        f"  correct rejections   {summary['rejection_rate']:.4f} "
        f"({summary['false_retrievals']} unanswerable question(s) retrieved "
        "something)"
    )
    print(
        f"  boundary retrieved   {summary['boundary_retrieved']}/"
        f"{summary['boundary']} (expected -- refusal is the safety layer's job)"
    )
    if "latency_ms" in summary:
        latency = summary["latency_ms"]
        print(
            f"  latency ms           mean {latency['mean']} "
            f"median {latency['median']} p95 {latency['p95']} max {latency['max']}"
        )
        print(
            f"    embed {summary['embed_ms_mean']} ms, "
            f"search {summary['search_ms_mean']} ms"
        )
    if summary["errors"]:
        print(f"  ERRORS               {summary['errors']}")


def run(top_k: int = 5, questions_path: Path | None = None) -> dict[str, Any]:
    """Evaluate retrieval and return the raw results plus the summary."""
    settings = get_settings()
    questions = load_questions(questions_path)
    retriever = Retriever(get_embedder(), get_vector_store(), settings)

    print(f"Evaluating retrieval over {len(questions)} question(s), top_k={top_k}")
    print(f"  collection      {settings.qdrant_collection}")
    print(f"  embedding model {settings.embedding_model}")
    print(f"  score threshold {settings.score_threshold}")

    # Load the embedding model before timing anything: otherwise the first
    # question absorbs several seconds of model load and the latency figures
    # describe a cold start rather than steady-state retrieval.
    warm_started = time.perf_counter()
    get_embedder().warm_up()
    warm_ms = round((time.perf_counter() - warm_started) * 1000, 2)
    print(f"  model warm-up   {warm_ms} ms (excluded from latency below)\n")

    results = [evaluate_question(retriever, item, top_k) for item in questions]
    summary = summarise(results, top_k)
    summary["warm_up_ms"] = warm_ms
    print_report(results, summary)

    return {"summary": summary, "results": [asdict(item) for item in results]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.evaluate_retrieval",
        description="Measure retrieval quality against the evaluation set.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the full run to a JSON file."
    )
    args = parser.parse_args(argv)

    configure_logging(level="WARNING", json_output=False)

    try:
        payload = run(top_k=args.top_k, questions_path=args.questions)
    except AIError as exc:
        print(f"\nERROR [{exc.code}] {exc.message}", file=sys.stderr)
        return 2

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
