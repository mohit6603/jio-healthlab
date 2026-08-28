---
title: SOP — Laboratory Workflow and Turnaround Times
category: sop
---

# SOP: Laboratory Workflow and Turnaround Times

## Purpose

To define the stages a request passes through and the turnaround commitments
attached to each priority, so that operational delay can be measured
consistently across branches.

## The report lifecycle

Every request in JIO HealthLab moves through a fixed sequence of statuses:

1. **Registered** — the request has been raised and an accession number issued;
   no sample has been collected yet.
2. **Collected** — the specimen has been taken and labelled at the collection
   point.
3. **Processing** — the sample has been received centrally, accepted, and is on
   or queued for an analyser.
4. **Review** — results are available and awaiting technical or clinical
   verification.
5. **Ready** — results have been verified and released for delivery.
6. **Delivered** — the report has been issued to the requester or patient.

A report may move backwards — for example from *processing* to *collected* —
when a re-collection is required. Every transition is timestamped.

## Turnaround time definitions

**Turnaround time (TAT)** is measured from the moment the sample is received at
the central laboratory to the moment the report reaches *ready*. Collection and
transport time is tracked separately as pre-analytical time, because it is
governed by courier scheduling rather than laboratory capacity.

## Priority categories

| Priority | Definition | Handling |
|---|---|---|
| Routine | Standard scheduling | Processed in receipt order within the workstation queue |
| Urgent | Clinically time-sensitive | Moved to the head of the queue; triggers unscheduled courier runs |

## Target turnaround by test group

These are the internal operational targets used for queue management and delay
reporting. They are commitments about laboratory throughput, not clinical
guarantees.

| Test group | Routine target | Urgent target |
|---|---|---|
| Haematology (CBC) | 4 hours | 1 hour |
| Routine chemistry (glucose, renal, liver, lipid) | 6 hours | 2 hours |
| Immunoassay (thyroid, hormones) | 24 hours | 4 hours |
| Urinalysis | 4 hours | 1 hour |
| Imaging (X-ray) | 12 hours | 2 hours |
| Imaging (MRI) | 48 hours | 12 hours |

## Causes of delay

The most common operational causes of a missed turnaround target are:

- **Queue depth** — the number of samples already waiting at a workstation when
  a new sample arrives.
- **Urgent load** — a high proportion of urgent samples displaces routine work
  further down the queue.
- **Analyser downtime** — scheduled calibration, maintenance or unplanned
  failure.
- **Sample rejection** — a rejected sample restarts the whole cycle including
  re-collection and transport.
- **Courier scheduling** — samples collected shortly after a scheduled run wait
  for the next one.
- **Reagent availability** — a batch awaiting reagent replenishment.
- **Verification backlog** — results sitting in *review* awaiting a verifier.

## Escalation

A request that is projected to exceed its target is escalated to the
workstation lead. Urgent requests projected to breach are escalated
immediately to the duty supervisor.
