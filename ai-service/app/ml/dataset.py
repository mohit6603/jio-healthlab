"""Synthetic dataset generator for report-delay prediction.

**This data is synthetic.** JIO HealthLab has no historical turnaround dataset
to train on, so one is generated from a documented causal story. Every metric
produced from it measures how well a model recovers *this generator*, not how
well it would predict delays in a real laboratory. Nothing here should be read
as a clinical or operational validation.

The generator is deliberately not trivial:

* The label is produced from a *latent* processing time, which is then thrown
  away. No feature encodes the outcome, so there is nothing to leak.
* Multiplicative lognormal noise puts an irreducible error floor in the data. A
  model scoring near 1.0 here would be evidence of a bug, not of skill.
* Turnaround targets match ``knowledge/sop/turnaround_and_workflow.md``, so the
  simulation and the documentation tell the same story.

Reproducible: the same seed always yields the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Bump when the generating process changes; recorded in model metadata so a
#: model can be traced to the data that produced it.
DATASET_VERSION = "synthetic-v1"

TARGET_COLUMN = "delayed"


@dataclass(frozen=True, slots=True)
class TestProfile:
    """Turnaround targets for one test type, in minutes."""

    group: str
    routine_tat: int
    urgent_tat: int
    #: Multiplier on baseline processing time; imaging is slower to report.
    effort: float


#: Targets mirror the turnaround SOP in the knowledge base.
TEST_PROFILES: dict[str, TestProfile] = {
    "CBC Panel": TestProfile("haematology", 240, 60, 1.0),
    "Blood Test": TestProfile("haematology", 240, 60, 1.0),
    "Sugar Test": TestProfile("chemistry", 360, 120, 0.9),
    "Cholesterol": TestProfile("chemistry", 360, 120, 1.0),
    "Liver Function": TestProfile("chemistry", 360, 120, 1.1),
    "Kidney Function": TestProfile("chemistry", 360, 120, 1.1),
    "Thyroid": TestProfile("immunoassay", 1440, 240, 1.6),
    "Urine Test": TestProfile("urinalysis", 240, 60, 0.8),
    "X-Ray": TestProfile("imaging", 720, 120, 1.3),
    "MRI Scan": TestProfile("imaging", 2880, 720, 2.2),
}


@dataclass(frozen=True, slots=True)
class BranchProfile:
    """Throughput characteristics of a collection branch."""

    city: str
    #: Relative daily volume. Busier branches queue more work.
    volume: float
    #: Capacity relative to volume. Below 1.0 means chronically under-resourced.
    capacity: float


BRANCH_PROFILES: dict[str, BranchProfile] = {
    "BKC Flagship": BranchProfile("Mumbai", 1.35, 1.25),
    "Andheri Hub": BranchProfile("Mumbai", 1.20, 0.90),
    "Aerocity Express": BranchProfile("Delhi", 1.10, 1.05),
    "Connaught Place": BranchProfile("Delhi", 1.00, 0.85),
    "Koregaon Park": BranchProfile("Pune", 0.85, 1.00),
    "SG Highway": BranchProfile("Ahmedabad", 0.80, 0.95),
    "Salt Lake": BranchProfile("Kolkata", 0.75, 0.80),
    "Whitefield": BranchProfile("Bengaluru", 1.05, 1.10),
}

#: Share of requests marked urgent.
URGENT_SHARE = 0.22

#: Relative intake by hour. Two peaks: morning collection and an evening batch.
_HOURLY_INTAKE = np.array(
    [
        0.20, 0.15, 0.12, 0.12, 0.18, 0.45, 0.90, 1.45,
        1.85, 1.80, 1.50, 1.20, 1.00, 0.95, 1.05, 1.25,
        1.55, 1.60, 1.30, 0.95, 0.70, 0.50, 0.35, 0.25,
    ]
)

#: Monday carries the weekend backlog; Sunday runs a skeleton shift.
_WEEKDAY_LOAD = np.array([1.25, 1.10, 1.05, 1.05, 1.15, 0.85, 0.60])

#: Probability an analyser is degraded when a request arrives.
_DOWNTIME_RATE = 0.07


def generate_dataset(n_samples: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset of laboratory requests.

    Returns one row per request with the features a scheduler could actually
    know at intake time, plus the binary ``delayed`` outcome.
    """
    rng = np.random.default_rng(seed)

    branches = list(BRANCH_PROFILES)
    branch_weights = np.array([BRANCH_PROFILES[b].volume for b in branches])
    branch_weights = branch_weights / branch_weights.sum()
    branch_idx = rng.choice(len(branches), size=n_samples, p=branch_weights)

    tests = list(TEST_PROFILES)
    # Routine panels dominate; imaging is comparatively rare.
    test_weights = np.array([1.6, 1.4, 1.3, 1.0, 0.9, 0.9, 1.0, 1.1, 0.5, 0.2])
    test_weights = test_weights / test_weights.sum()
    test_idx = rng.choice(len(tests), size=n_samples, p=test_weights)

    hour_p = _HOURLY_INTAKE / _HOURLY_INTAKE.sum()
    created_hour = rng.choice(24, size=n_samples, p=hour_p)
    created_dow = rng.integers(0, 7, size=n_samples)

    is_urgent = rng.random(n_samples) < URGENT_SHARE

    branch_volume = np.array([BRANCH_PROFILES[branches[i]].volume for i in branch_idx])
    branch_capacity = np.array(
        [BRANCH_PROFILES[branches[i]].capacity for i in branch_idx]
    )
    effort = np.array([TEST_PROFILES[tests[i]].effort for i in test_idx])

    # --- observable state at intake -------------------------------------
    intake_pressure = (
        _HOURLY_INTAKE[created_hour] * _WEEKDAY_LOAD[created_dow] * branch_volume
    )
    queue_size = rng.poisson(np.clip(intake_pressure * 11.0, 1.0, None))
    current_workload = rng.poisson(np.clip(queue_size * 2.4 + 4.0, 1.0, None))
    urgent_report_count = rng.binomial(queue_size, URGENT_SHARE)
    analyser_degraded = rng.random(n_samples) < _DOWNTIME_RATE

    expected_tat = np.array(
        [
            TEST_PROFILES[tests[i]].urgent_tat
            if urgent
            else TEST_PROFILES[tests[i]].routine_tat
            for i, urgent in zip(test_idx, is_urgent, strict=True)
        ],
        dtype=float,
    )

    # Historical average for this branch/test pairing, as a scheduler would
    # have it: centred on the target, shifted by how well-resourced the branch
    # is, with per-request measurement noise.
    # Typical processing sits comfortably inside the target when the lab is
    # quiet; delay comes from pressure, not from a baseline that already
    # misses. Effort is folded in here rather than applied again below --
    # a historical average already reflects how slow the test is to report.
    historical_avg_tat = (
        expected_tat
        * (0.50 / branch_capacity)
        * (1.0 + 0.10 * (effort - 1.0))
        * rng.lognormal(0.0, 0.10, n_samples)
    )

    # --- latent processing time (never becomes a feature) ---------------
    queue_pressure = 1.0 + 0.026 * queue_size / np.maximum(branch_capacity, 0.5)
    urgent_pressure = 1.0 + 0.055 * urgent_report_count
    # Requests arriving late in the day wait for the next shift.
    handover_penalty = 1.0 + 0.28 * ((created_hour >= 18) | (created_hour < 6))
    downtime_penalty = np.where(analyser_degraded, 1.0 + rng.uniform(0.3, 1.1, n_samples), 1.0)
    # Urgent work is expedited, but its target is far tighter.
    expedite = np.where(is_urgent, 0.62, 1.0)

    actual_tat = (
        historical_avg_tat
        * queue_pressure
        * urgent_pressure
        * handover_penalty
        * downtime_penalty
        * expedite
        * rng.lognormal(0.0, 0.32, n_samples)  # irreducible noise
    )

    delayed = (actual_tat > expected_tat).astype(int)

    frame = pd.DataFrame(
        {
            "test_type": [tests[i] for i in test_idx],
            "test_group": [TEST_PROFILES[tests[i]].group for i in test_idx],
            "priority": np.where(is_urgent, "urgent", "routine"),
            "branch": [branches[i] for i in branch_idx],
            "city": [BRANCH_PROFILES[branches[i]].city for i in branch_idx],
            "created_hour": created_hour.astype(int),
            "created_day_of_week": created_dow.astype(int),
            "expected_tat_minutes": expected_tat.round(1),
            "historical_avg_tat_minutes": historical_avg_tat.round(1),
            "queue_size": queue_size.astype(int),
            "current_workload": current_workload.astype(int),
            "urgent_report_count": urgent_report_count.astype(int),
            "analyser_degraded": analyser_degraded.astype(int),
            TARGET_COLUMN: delayed,
        }
    )
    return frame


def describe_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Summary recorded alongside a trained model."""
    return {
        "dataset_version": DATASET_VERSION,
        "synthetic": True,
        "rows": len(frame),
        "positive_rate": round(float(frame[TARGET_COLUMN].mean()), 4),
        "columns": list(frame.columns),
    }
