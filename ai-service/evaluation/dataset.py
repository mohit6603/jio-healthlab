"""Loading the evaluation question set."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "questions.json"

#: Kinds where retrieval itself should find nothing: the knowledge base
#: genuinely has no answer.
NO_RETRIEVAL_KINDS = frozenset({"out_of_domain", "prompt_injection"})

#: Clinical-boundary questions DO mention real topics -- haemoglobin, TSH,
#: cholesterol -- so retrieval is expected to find relevant chunks. Refusing
#: them is the safety layer's job, before generation, not retrieval's. Scoring
#: them as retrieval failures would measure the wrong component.
BOUNDARY_KINDS = frozenset({"clinical_boundary"})

#: Everything that should not produce a normal grounded answer.
NEGATIVE_KINDS = NO_RETRIEVAL_KINDS | BOUNDARY_KINDS


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    question: str
    expected_source: str | None
    expected_keywords: tuple[str, ...] = field(default_factory=tuple)
    kind: str = "factual"

    @property
    def is_negative(self) -> bool:
        return self.kind in NEGATIVE_KINDS

    @property
    def expects_no_retrieval(self) -> bool:
        return self.kind in NO_RETRIEVAL_KINDS

    @property
    def is_boundary(self) -> bool:
        return self.kind in BOUNDARY_KINDS


def load_questions(path: Path | None = None) -> list[EvalQuestion]:
    """Read the question set from disk."""
    data = json.loads((path or QUESTIONS_PATH).read_text(encoding="utf-8"))
    return [
        EvalQuestion(
            id=item["id"],
            question=item["question"],
            expected_source=item.get("expected_source"),
            expected_keywords=tuple(item.get("expected_keywords", [])),
            kind=item.get("kind", "factual"),
        )
        for item in data["questions"]
    ]
