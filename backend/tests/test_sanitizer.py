"""PII boundary tests.

The central assertion is negative: given a report stuffed with identifying
data, none of it may appear in what leaves the backend.
"""

from datetime import datetime, timedelta

import pytest

from app.models import Report
from app.services.sanitizer import (
    EXCLUDED_FIELDS,
    age_band,
    sanitize_report,
)


def make_report(**overrides) -> Report:
    defaults = {
        "id": 42,
        "patient_name": "Asha Nair",
        "age": 34,
        "gender": "Female",
        "phone": "+91 98765 10001",
        "email": "asha.nair@example.com",
        "city": "Mumbai",
        "lab_branch": "BKC Flagship",
        "test_type": "CBC Panel",
        "doctor_name": "Dr. Naina Rao",
        "status": "processing",
        "priority": "urgent",
        "notes": "Call patient on 9876543210 before releasing. Sister is Priya.",
        "result_due_at": None,
    }
    return Report(**{**defaults, **overrides})


# ------------------------------------------------------------ exclusions ---
IDENTIFIERS = [
    "Asha Nair",
    "asha.nair@example.com",
    "+91 98765 10001",
    "9876543210",
    "Dr. Naina Rao",
    "Priya",
    "Female",
]


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_no_identifier_survives_sanitisation(identifier):
    rendered = sanitize_report(make_report()).render()

    assert identifier not in rendered


def test_exact_age_is_not_sent():
    rendered = sanitize_report(make_report(age=34)).render()

    assert "34" not in rendered
    assert "30-39" in rendered


def test_report_id_is_not_sent():
    """The id is a handle back to the full record; it does not belong in a prompt."""
    rendered = sanitize_report(make_report(id=42)).render()

    assert "42" not in rendered


def test_notes_are_excluded_wholesale():
    fields = sanitize_report(make_report()).fields

    assert "notes" not in fields


def test_excluded_fields_never_appear_as_keys():
    fields = sanitize_report(make_report()).fields

    assert EXCLUDED_FIELDS.isdisjoint(fields)


def test_sanitised_output_is_an_allow_list():
    """A new column on Report must not appear without a deliberate change."""
    fields = set(sanitize_report(make_report()).fields)

    assert fields <= {
        "test_type",
        "status",
        "priority",
        "city",
        "branch",
        "age_group",
        "due_status",
    }


# -------------------------------------------------------------- contents ---
def test_operational_fields_are_kept():
    fields = sanitize_report(make_report()).fields

    assert fields["test_type"] == "CBC Panel"
    assert fields["status"] == "processing"
    assert fields["priority"] == "urgent"
    assert fields["city"] == "Mumbai"
    assert fields["branch"] == "BKC Flagship"


def test_search_text_is_the_test_name():
    assert sanitize_report(make_report()).search_text == "CBC Panel"


def test_render_is_key_value_lines():
    rendered = sanitize_report(make_report()).render()

    assert "test_type: CBC Panel" in rendered
    assert "priority: urgent" in rendered


def test_missing_optional_fields_are_omitted():
    fields = sanitize_report(make_report(city=None, lab_branch=None)).fields

    assert "city" not in fields
    assert "branch" not in fields


# ------------------------------------------------------------- age bands ---
@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (0, "0-9"),
        (9, "0-9"),
        (10, "10-19"),
        (34, "30-39"),
        (79, "70-79"),
        (80, "80+"),
        (104, "80+"),
    ],
)
def test_age_bands(age, expected):
    assert age_band(age) == expected


def test_unknown_age_band():
    assert age_band(None) == "unknown"


# ------------------------------------------------------------ due status ---
def test_overdue_request_is_flagged():
    now = datetime(2026, 6, 1, 12, 0)
    report = make_report(result_due_at=now - timedelta(hours=5), status="processing")

    assert sanitize_report(report, now=now).fields["due_status"] == "overdue"


def test_completed_request_is_not_overdue():
    now = datetime(2026, 6, 1, 12, 0)
    report = make_report(result_due_at=now - timedelta(hours=5), status="delivered")

    assert sanitize_report(report, now=now).fields["due_status"] == "on schedule"


def test_future_due_date_is_on_schedule():
    now = datetime(2026, 6, 1, 12, 0)
    report = make_report(result_due_at=now + timedelta(hours=5))

    assert sanitize_report(report, now=now).fields["due_status"] == "on schedule"


def test_no_due_date_omits_the_field():
    assert "due_status" not in sanitize_report(make_report()).fields


def test_exact_due_timestamp_is_not_sent():
    now = datetime(2026, 6, 1, 12, 0)
    report = make_report(result_due_at=now + timedelta(hours=5))

    rendered = sanitize_report(report, now=now).render()

    assert "2026" not in rendered
    assert "17:00" not in rendered
