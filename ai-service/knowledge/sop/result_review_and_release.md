---
title: SOP — Result Review, Verification and Release
category: sop
---

# SOP: Result Review, Verification and Release

## Purpose

To ensure every released result has been checked for analytical validity and
that results requiring urgent clinical attention are communicated without
waiting for the routine reporting cycle.

## Scope

Applies to laboratory technologists and authorised verifiers.

## Technical review

Before a result leaves *review*, the verifier confirms that:

- the run's quality-control samples were within their acceptance limits;
- the analyser raised no flags on the sample — for example haemolysis,
  lipaemia, icterus or a clot detection alarm;
- the result is internally consistent with the rest of the panel;
- where a previous result exists, the change over time is plausible.

## Delta checking

A **delta check** compares a result against the same patient's previous result
for that analyte. A change larger than the configured threshold is held for
review rather than released automatically. A failed delta check does not mean
the result is wrong — it means it warrants a second look before release.

## Repeat and dilution rules

Results outside the analyser's measuring range are repeated on dilution.
Implausible results are repeated on the original sample before a re-collection
is requested, since analytical error is more common than a true extreme value.

## Critical results

A **critical result** is one that may require immediate clinical action. These
are communicated by direct contact with the requester rather than through the
routine report, and the communication itself is logged: who was contacted, by
whom, at what time, and confirmation that the value was read back correctly.

Critical-result handling proceeds in parallel with normal release — it does not
wait for the report to reach *ready*.

## Release

Once verified, the result moves to *ready* and becomes available to the
requester. The verifier's identity and the release timestamp are recorded
against the accession number.

## Amendments

A released report that later requires correction is amended rather than
overwritten. The amended report is clearly marked as such, the original value
is retained in the audit trail, and the requester is notified.

## Audit trail

Every status transition, verification, critical-result call and amendment is
recorded with the acting user and a timestamp. The audit trail is append-only.
