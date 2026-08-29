"""Log redaction tests.

Structured logs are shipped to CloudWatch and read by operators. Patient
identifiers must not be in them, whether they arrive as a field name or
embedded in free text.
"""

import json
import logging

import pytest

from app.core.logging import REDACTED, JsonFormatter, redact


def render(message: str, **extra) -> dict:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


# ------------------------------------------------------------ field names --
@pytest.mark.parametrize(
    "field",
    ["patient_name", "phone", "email", "password", "token", "api_key", "prompt"],
)
def test_sensitive_field_names_are_redacted(field):
    assert render("event", **{field: "sensitive value"})[field] == REDACTED


def test_sensitive_names_are_matched_case_insensitively():
    assert redact({"Patient_Name": "Asha Nair"})["Patient_Name"] == REDACTED


def test_operational_fields_are_kept():
    payload = render("event", report_id=42, status=200, latency_ms=12.5)

    assert payload["report_id"] == 42
    assert payload["status"] == 200
    assert payload["latency_ms"] == 12.5


# ------------------------------------------------------------- free text ---
def test_email_in_free_text_is_redacted():
    assert "asha@example.com" not in json.dumps(
        render("user asha@example.com signed in")
    )


def test_phone_number_in_free_text_is_redacted():
    assert "9876543210" not in json.dumps(render("call 9876543210 back"))


def test_bearer_token_is_redacted():
    payload = render("auth header Bearer abc123.def456")

    assert "abc123.def456" not in json.dumps(payload)


# ------------------------------------------------------------- nesting ----
def test_nested_structures_are_redacted():
    cleaned = redact({"outer": {"inner": {"email": "a@b.com"}}})

    assert cleaned["outer"]["inner"]["email"] == REDACTED


def test_lists_are_redacted():
    cleaned = redact([{"phone": "+91 98765 10001"}, {"status": "ok"}])

    assert cleaned[0]["phone"] == REDACTED
    assert cleaned[1]["status"] == "ok"


def test_non_string_values_survive():
    cleaned = redact({"count": 3, "ratio": 0.5, "flag": True, "nothing": None})

    assert cleaned == {"count": 3, "ratio": 0.5, "flag": True, "nothing": None}


# ------------------------------------------------ end-to-end through app ---
def test_report_creation_does_not_log_the_patient_name(client, caplog):
    caplog.set_level(logging.INFO)

    client.post(
        "/api/reports",
        json={
            "patient_name": "Asha Nair",
            "age": 34,
            "test_type": "CBC Panel",
            "email": "asha.nair@example.com",
            "phone": "+91 98765 10001",
        },
    )

    logged = "\n".join(
        JsonFormatter().format(record) for record in caplog.records
    )
    assert "Asha Nair" not in logged
    assert "asha.nair@example.com" not in logged
