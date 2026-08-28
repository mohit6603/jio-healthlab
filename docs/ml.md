# Report delay prediction

Predicting whether a laboratory request will miss its turnaround target.

> **The training data is synthetic.** Every metric here measures how well a
> model recovers a simulation, not how it would perform in a real laboratory.
> Nothing on this page is a clinical or operational validation.

---

## Why synthetic

There is no historical turnaround dataset. The options were to ship no ML, to
fabricate results, or to build a documented generator and be explicit that the
numbers describe it. The third is the only honest one that still demonstrates
a working pipeline.

`app/ml/dataset.py` generates requests from a causal story whose turnaround
targets match `knowledge/sop/turnaround_and_workflow.md`, so the simulation and
the documentation agree.

The label comes from a **latent** processing time that is then discarded:

```
actual = historical_avg
       × queue_pressure        (1 + 0.026 × queue / branch_capacity)
       × urgent_pressure       (1 + 0.055 × urgent_ahead)
       × handover_penalty      (evening and night arrivals)
       × downtime_penalty      (degraded analyser)
       × expedite              (urgent work is prioritised)
       × lognormal(0, 0.32)    (irreducible noise)

delayed = actual > expected_tat
```

No feature encodes `actual`. The lognormal term guarantees an error floor — a
model scoring near 1.0 here would be evidence of a bug, not skill.

### Tuning it took a search

The first constants produced a **65 %** delay rate — a laboratory missing two
thirds of its targets. The second produced **9.7 %** with almost no queue
signal. A parameter sweep landed on **32.4 %** with clean monotone structure:

| Queue quartile | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Delay rate | 0.19 | 0.28 | 0.37 | 0.50 |

Degraded analyser: 0.29 → 0.73. The three worst branches are exactly the three
lowest-capacity ones.

---

## Features

Everything is knowable at intake. Nothing is derived from the outcome.

**Categorical** — `test_type`, `test_group`, `priority`, `branch`, `city`

**Numeric** — `created_hour`, `created_day_of_week`, `expected_tat_minutes`,
`historical_avg_tat_minutes`, `queue_size`, `current_workload`,
`urgent_report_count`, `analyser_degraded`

**Derived** — `tat_headroom_ratio` (historical ÷ target; near 1.0 means no
slack), `urgent_share_of_queue`, `off_hours_arrival`

The feature list is stored with every artifact and validated at inference, so a
model cannot be served with different columns from the ones it learned.

---

## Training

```bash
docker compose exec ai-service python -m app.ml.train
docker compose exec ai-service python -m app.ml.train --samples 30000 --seed 7
docker compose exec ai-service python -m app.ml.train --dry-run
```

```
generate → validate → engineer → split → train → compare
         → select → evaluate → persist
```

**Three splits, not two.** Candidates are compared on a validation split; the
reported metrics come from a test split that selection never saw. Choosing on
the data you report from inflates the numbers.

---

## Measured results

12,000 rows, seed 42, held-out test split.

| Candidate | Validation ROC-AUC |
|---|---|
| **logistic_regression** | **0.8561** ← selected |
| gradient_boosting | 0.8477 |
| random_forest | 0.8432 |

```
accuracy 0.7762   precision 0.6193   recall 0.8018   f1 0.6988
roc_auc  0.8639   average_precision 0.7459   brier 0.1539
confusion  tn=1240  fp=383  fn=154  tp=623
```

Logistic regression winning is consistent with the generator being
multiplicative in its features — a log-linear model is close to the truth.

### Leakage check

Retraining on **shuffled labels** gives ROC-AUC **0.4843** — chance — against
0.8639 on real labels. A leaking feature would keep that high.

The fitted coefficients also recover the generator's mechanism in the right
direction and rough order:

| Feature | Coefficient |
|---|---|
| `priority_urgent` | −1.55 |
| `tat_headroom_ratio` | +0.99 |
| `queue_size` | +0.77 |
| `analyser_degraded` | +0.72 |
| `urgent_report_count` | +0.58 |
| `off_hours_arrival` | +0.52 |
| `branch_BKC Flagship` (highest capacity) | −0.40 |

**XGBoost was considered and left out.** It adds a compiled dependency to an
already 2 GB image and occupies the same niche as scikit-learn's gradient
boosting at this data size.

---

## Serving

```bash
curl -X POST localhost:8001/ml/predict-delay \
  -H 'Content-Type: application/json' \
  -d '{"test_type":"CBC Panel","priority":"urgent","branch":"Andheri Hub"}'
```

Only `test_type` is required. A scheduler knows the test and the branch, not
the branch's historical average turnaround, so missing values are filled from
the same profiles the training data was built from — and **every filled value
is returned in `inferred_fields`**, so callers can see which numbers were
theirs.

### Risk bands

```
low     p < 0.35
medium  0.35 ≤ p < 0.65
high    p ≥ 0.65
```

Deterministic, defined once in `ml/predictor.py`, and the OpenAPI description
is generated from the same constants so documentation cannot drift. Set
relative to the 0.32 base rate: "low" is at or below it, "high" roughly double.

They are a presentation choice, not a property of the model.

---

## Versioning

Every artifact is saved with a sidecar metadata document:

```
model_name, version, algorithm, trained_at, dataset version + seed,
feature schema, test metrics, every candidate's validation score,
selection metric, synthetic_data flag
```

Versions are timestamped (`v20260828074424`) — sortable and unique. Every
prediction response carries `model_version`, so a stored prediction traces back
to its artifact, its metrics and its training data.

`GET /ml/model` exposes all of it.

Artifacts live in the `model-artifacts` volume and are **never committed**.
Without that volume a `docker compose up --build` silently discards the model
and every prediction endpoint returns `ML_MODEL_NOT_TRAINED`.

---

## MLflow

Not used. The metadata sidecar already records parameters, metrics, artifact
path and version, and `GET /ml/model` serves it. Adding an MLflow server would
mean another container and another dependency for a single model with one
training entry point. Worth revisiting with multiple models or real experiment
sweeps.
