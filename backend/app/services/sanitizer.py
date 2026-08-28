"""PII boundary for report data leaving the backend.

The AI service is a separate process that talks to a vector database and a
language model. Nothing that could identify a patient may cross into it, so the
boundary is an allow-list, not a deny-list: a field is excluded unless it has
been explicitly judged safe. Adding a column to ``Report`` therefore cannot
silently start leaking it.

What crosses:  test type, status, priority, city, branch, age band, and whether
               the request is overdue.
What does not: patient name, phone, email, doctor name, free-text notes, exact
               dates of birth or ages, and the report id.

Notes are excluded wholesale. They are free text written by staff and can
contain anything -- a callback number, a relative's name -- and no pattern
scrub can be trusted with that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..models import Report

#: Fields that must never reach the AI service. Asserted by the tests.
EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "patient_name",
        "phone",
        "email",
        "doctor_name",
        "notes",
        "age",
        "gender",
        "id",
        "sample_collected_at",
        "result_due_at",
        "created_at",
        "updated_at",
    }
)

#: Ten-year bands. Coarse enough not to single anyone out, useful enough that
#: a paediatric or geriatric panel reads differently from an adult one.
_AGE_BAND_WIDTH = 10
_MAX_BANDED_AGE = 80


@dataclass(slots=True)
class SanitizedReport:
    """Non-identifying representation of a laboratory request."""

    #: Ordered, human-readable key/value pairs safe to send onward.
    fields: dict[str, str] = field(default_factory=dict)
    #: What the knowledge base is searched with -- the test name.
    search_text: str = ""

    def render(self) -> str:
        """Render as the plain text handed to the AI service."""
        return "\n".join(f"{key}: {value}" for key, value in self.fields.items())


def age_band(age: int | None) -> str:
    """Bucket an age into a ten-year band."""
    if age is None or age < 0:
        return "unknown"
    if age >= _MAX_BANDED_AGE:
        return f"{_MAX_BANDED_AGE}+"
    lower = (age // _AGE_BAND_WIDTH) * _AGE_BAND_WIDTH
    return f"{lower}-{lower + _AGE_BAND_WIDTH - 1}"


def sanitize_report(report: Report, *, now: datetime | None = None) -> SanitizedReport:
    """Build the non-identifying view of ``report``.

    Only the fields named here are ever included; everything else on the model
    is left behind by construction.
    """
    moment = now or datetime.now(UTC).replace(tzinfo=None)

    fields: dict[str, str] = {"test_type": report.test_type}
    if report.status:
        fields["status"] = report.status
    if report.priority:
        fields["priority"] = report.priority
    if report.city:
        fields["city"] = report.city
    if report.lab_branch:
        fields["branch"] = report.lab_branch
    fields["age_group"] = age_band(report.age)

    if report.result_due_at is not None:
        due = report.result_due_at
        overdue = due < moment and report.status not in {"ready", "delivered"}
        fields["due_status"] = "overdue" if overdue else "on schedule"

    return SanitizedReport(fields=fields, search_text=report.test_type)


#: Payload keys the report index may filter on. Kept in step with the AI
#: service's FILTERABLE_FIELDS.
#: Trailing words that already make the test name read as a noun phrase.
_TEST_NOUNS = frozenset({"test", "scan", "panel", "profile", "screen", "x-ray"})

SEARCH_FILTER_FIELDS: tuple[str, ...] = (
    "status",
    "priority",
    "branch",
    "city",
    "test_type",
)


def searchable_text(report: Report) -> str:
    """Build the sentence that represents a report in the semantic index.

    Written as prose rather than key/value pairs because it is embedded: a
    sentence sits closer in vector space to a natural-language query like
    "urgent kidney tests waiting in Mumbai" than a field dump does.

    Contains only operational fields. Nothing identifying is ever embedded.
    """
    priority = (report.priority or "routine").capitalize()
    # "Blood Test" already ends in a noun; appending "test" reads badly, and
    # this string is embedded, so it should read like natural language.
    name = report.test_type
    suffix = "" if name.lower().split()[-1] in _TEST_NOUNS else " test"
    parts = [f"{priority} {name}{suffix}"]

    if report.lab_branch and report.city:
        parts.append(f"at the {report.lab_branch} branch in {report.city}")
    elif report.lab_branch:
        parts.append(f"at the {report.lab_branch} branch")
    elif report.city:
        parts.append(f"in {report.city}")

    sentence = " ".join(parts) + "."

    extras = [f"Status: {report.status}." if report.status else ""]
    if report.priority == "urgent":
        extras.append("Marked urgent and prioritised in the queue.")
    extras.append(f"Patient age group {age_band(report.age)}.")

    return " ".join(part for part in [sentence, *extras] if part)


def search_metadata(report: Report) -> dict[str, str]:
    """Exact-match filter keys for one report."""
    values = {
        "status": report.status,
        "priority": report.priority,
        "branch": report.lab_branch,
        "city": report.city,
        "test_type": report.test_type,
    }
    return {key: str(value) for key, value in values.items() if value}
