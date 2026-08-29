# Security

What is enforced, how, and — importantly — what is not.

---

## Authentication

**Passwords** — argon2id via `argon2-cffi`, used directly rather than through
passlib, which has had no release in years and has known bcrypt
incompatibilities. Hashes are transparently upgraded on login when cost
parameters move on. Verification returns `False` for a corrupt hash rather than
raising, so hashing internals cannot surface as a 500.

**Sessions** — a 15-minute access JWT plus a revocable refresh token. Only a
SHA-256 digest of the refresh token is stored: a database leak alone cannot be
replayed as a session.

**Rotation with reuse detection** — every refresh issues a new token and
revokes the old one. Presenting an already-revoked token revokes **every**
session for that user. Replay is indistinguishable from theft, and it is not
possible to tell whether the legitimate user or an attacker holds the current
token, so ending all of them is the only safe response.

**No account enumeration** — an unknown email and a wrong password return
byte-identical responses, and an unknown email still pays for a hash so timing
does not leak either. Lockout does not leak it either: an unknown email never
produces the lockout message, only `INVALID_CREDENTIALS`.

**Account lockout** — 8 consecutive failures locks an account for 15 minutes.
Checked *before* the password, so a locked account cannot be probed by
continuing to guess, and a correct password is refused while locked. The
counter resets on any successful sign-in, so the threshold is consecutive
failures rather than lifetime ones. Time-based rather than permanent: a
legitimate user recovers without an administrator, and an attacker gains only
a delay. `auth_service.unlock_user` clears it manually.

**Session storage** — the refresh token is an **httpOnly, SameSite=Strict
cookie** scoped to `/api/auth`. JavaScript cannot read it, so an XSS running in
the page cannot exfiltrate a session. Verified in a real browser while signed
in: `localStorage`, `sessionStorage` and `document.cookie` are all empty, and a
full page reload still restores the session. `Secure` is set automatically in
production and off locally, where plain http would make the browser discard the
cookie. The access token stays in memory for the tab's lifetime.

A request body is still accepted for non-browser clients that cannot hold
cookies; when both are present the cookie wins, so a client cannot downgrade
to an older token.

**Production guard** — the application refuses to start with
`ENVIRONMENT=production` while `JWT_SECRET_KEY` or `SEED_ADMIN_PASSWORD` is the
development placeholder. A placeholder signing key in production means anyone
can mint a token for any role.

---

## Authorisation

Routes declare a **permission**; the role that holds it is decided in
`security/permissions.py` and nowhere else. No handler compares role strings.

| Role | Can |
|---|---|
| `VIEWER` | Read reports and dashboard; informational AI only |
| `LAB_TECH` | + create/update reports, risk analytics, report search |
| `DOCTOR` | + report explanation, report search — no pipeline writes |
| `ADMIN` | Everything: users, ingestion, index rebuilds, audit, metrics |

`GET /api/auth/me` returns the permission list so a client can hide what it
cannot do. A test asserts that list matches what the server enforces for every
role — the client-side check is convenience, never the boundary.

---

## PII

An **allow-list**, not a deny-list: a field is excluded unless explicitly
judged safe, so adding a column to `Report` cannot silently start leaking it.

```
crosses to AI   test_type, status, priority, city, branch, age band, due status
never leaves    patient_name, phone, email, doctor_name, notes,
                exact age, gender, report id, exact timestamps
```

Notes are dropped whole rather than scrubbed. They are free text staff write —
a callback number, a relative's name — and no pattern match should be trusted
with that.

**Verified, not asserted.** With a report containing a name, phone, email,
doctor and a note carrying a second phone number and a relative's name:
`context_sent` was `{test_type, status, priority, city, branch, age_group}`,
and grepping both container log streams and the entire `healthlab_reports`
Qdrant collection for all seven identifiers found none.

**Logs** — structured JSON with central redaction of sensitive field names,
emails, phone numbers and bearer tokens, in nested structures and lists.

**Audit** — report updates record field *names*, not values. A failed login
does not record the attempted email: it is unverified input, and recording it
fills the trail with whatever anyone types. `ai_query_logs` has **no column**
that could hold a question or an answer.

---

## Healthcare boundary

Enforced in code, because the prompt was not enough.

Measured: asked *"My haemoglobin is 9. Do I have anaemia? What treatment should
I take?"*, Qwen2.5-0.5B-Instruct — with a system prompt that said "do not
diagnose" — replied *"your haemoglobin level of 9 mg/dL indicates mild
anemia"*, recommended iron supplements and blood transfusions, and invented the
unit. None of it was in the retrieved context.

So the boundary no longer depends on the model:

1. Questions asking for a diagnosis, a personal result interpretation or
   treatment advice are answered with a fixed redirect and **never reach the
   model** (verified: `generation_ms = 0`).
2. Generated answers are re-screened. A clinical assertion that slips through
   replaces the answer *and drops the citations*, which would otherwise lend it
   authority.

### Limits

These are **regex heuristics**. They will miss phrasings and they can
over-block — one false positive was found by the evaluation suite and fixed
(the bare phrase "you have" matched "a sample you have already given").

Both directions are tested: 14 clinical requests blocked, 14 informational
questions allowed, 8 unsafe answers caught, 8 safe answers not flagged, plus 12
regression cases from the false positive.

**This reduces risk. It does not make the system safe for clinical use.** No
regulatory claim is made — not HIPAA compliance, not FDA clearance, not
clinical validation. It is an engineering demonstration on synthetic data.

---

## Prompt injection

Knowledge documents are administrator-uploaded and, in a real deployment, could
be tampered with. Retrieved text is therefore treated as untrusted:

- fenced between explicit `<<<REFERENCE_MATERIAL` delimiters;
- the system prompt states the content is **data, not instructions**, and that
  embedded instructions must be ignored;
- a test asserts injected text lands **inside** the fence rather than in the
  instruction region;
- the evaluation suite includes an injection question and measures system
  prompt leaks — currently **0**.

### Limits — read this

**None of this makes prompt injection impossible.** Delimiters and
instructions are mitigation, not a security boundary: a sufficiently capable
model can still be talked out of its instructions by content inside the fence,
and a smaller model may simply comply.

What genuinely limits the blast radius is that the model has **no capability to
abuse**: no tools, no function calling, no database access, no ability to make
network requests. The worst a successful injection achieves is a wrong or
embarrassing answer in one response. It cannot exfiltrate data it was never
given — and the PII boundary means it is never given patient identifiers.

The defences that would matter if the model gained tools — capability scoping,
human confirmation for side effects, output filtering on egress — are not
implemented, because the model has no tools.

---

## Transport and input

- **HTTPS** terminated by host nginx; HSTS once a real certificate is in place.
- **Security headers** — CSP, `X-Frame-Options: DENY`, `nosniff`,
  `Referrer-Policy`, `Permissions-Policy`, `server_tokens off`. Verified on a
  live response, after finding that nginx silently drops server-level
  `add_header` directives in any location that declares its own.
- **CORS** — an explicit allow-list from `BACKEND_CORS_ORIGINS`, never `*`.
- **Uploads** — extension, MIME type and size validated before parsing;
  12 MB ceiling at nginx and 10 MB in the service. No arbitrary file execution.
- **SQL** — SQLAlchemy parameterised throughout; no string-built SQL anywhere.
- **Errors** — a single `{"error": {"code", "message"}}` envelope. Tracebacks
  are never returned; in production the message for an unhandled exception is
  deliberately generic and the detail goes to the log keyed by `request_id`.

### Rate limiting

At the host nginx, not in the application, so an attacker never reaches Python:

| Scope | Limit | Why |
|---|---|---|
| `/api/auth/login` | 5/min, burst 3 | The endpoint worth brute forcing |
| AI generation | 20/min, burst 5 | Each request holds a worker for tens of seconds |
| Everything else | 30/s, burst 50 | Normal browsing |

Config in `deploy/nginx/healthlab.conf`.

---

## Network exposure

In production only the web tier is published. The API, the AI service and
Qdrant stay on the internal network — Qdrant has **no authentication** in this
configuration and must never be reachable off-host. Locally every service binds
to `127.0.0.1`, not `0.0.0.0`.

Containers run unprivileged: backend uid 10001, AI service uid 10002, nginx
uid 101 on port 8080.

---

## Known gaps

Named rather than glossed over.

| Gap | Impact | Would fix |
|---|---|---|
| No MFA | Password alone is the factor | TOTP for `ADMIN` |
| No password rotation policy | A weak password stays valid indefinitely | Expiry and history checks |
| Secrets via environment | Visible to anything reading the process | AWS Secrets Manager with rotation |
| Audit trail append-only by convention | A DB admin could edit history | Append-only storage or off-host shipping |
| No per-user AI quota | One user could monopolise generation | Per-user token budget |
| Qdrant unauthenticated | Full read/write to anything on the network | API key, enforced by network policy today |

---

## Dependency scanning

`pip-audit` (both Python services) and `npm audit --audit-level=high` run as
**CI gates**, not advisory steps: a high-severity advisory fails the build.

They run in CI rather than only locally because a transitive dependency can
become vulnerable without this repository changing at all.

The first run found three high-severity advisories in `vite` — a `server.fs.deny`
bypass and an NTLMv2 hash disclosure via UNC path handling. Patched; the audit
is clean and the gate now prevents a regression.

---

## Reporting

This is a portfolio project on synthetic data with no real patient information.
For a genuine issue, open an issue on the repository.
