"""Synthetic dataset generator tests.

The generator is the foundation of every ML metric in this project, so its
properties are asserted rather than assumed: reproducibility, absence of
leakage, and a signal that matches the documented causal story.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset import (
    BRANCH_PROFILES,
    DATASET_VERSION,
    TARGET_COLUMN,
    TEST_PROFILES,
    describe_dataset,
    generate_dataset,
)


@pytest.fixture(name="frame", scope="module")
def frame_fixture() -> pd.DataFrame:
    return generate_dataset(6000, seed=42)


# ------------------------------------------------------- reproducibility ---
def test_same_seed_gives_identical_data():
    assert generate_dataset(500, seed=7).equals(generate_dataset(500, seed=7))


def test_different_seed_gives_different_data():
    assert not generate_dataset(500, seed=7).equals(generate_dataset(500, seed=8))


def test_row_count_is_respected():
    assert len(generate_dataset(123, seed=1)) == 123


# ---------------------------------------------------------------- shape ----
def test_target_is_binary(frame):
    assert set(frame[TARGET_COLUMN].unique()) == {0, 1}


def test_no_missing_values(frame):
    assert not frame.isna().any().any()


def test_class_balance_is_learnable(frame):
    """Neither so rare nor so common that the problem is degenerate."""
    assert 0.15 < frame[TARGET_COLUMN].mean() < 0.50


def test_categorical_values_come_from_the_profiles(frame):
    assert set(frame["test_type"]).issubset(TEST_PROFILES)
    assert set(frame["branch"]).issubset(BRANCH_PROFILES)


def test_city_matches_its_branch(frame):
    for branch, city in frame[["branch", "city"]].drop_duplicates().itertuples(
        index=False
    ):
        assert BRANCH_PROFILES[branch].city == city


def test_expected_tat_matches_the_documented_targets(frame):
    for row in frame.sample(200, random_state=0).itertuples():
        profile = TEST_PROFILES[row.test_type]
        expected = (
            profile.urgent_tat if row.priority == "urgent" else profile.routine_tat
        )
        assert row.expected_tat_minutes == pytest.approx(expected)


def test_urgent_count_never_exceeds_the_queue(frame):
    assert (frame["urgent_report_count"] <= frame["queue_size"]).all()


def test_hours_and_days_are_in_range(frame):
    assert frame["created_hour"].between(0, 23).all()
    assert frame["created_day_of_week"].between(0, 6).all()


# --------------------------------------------------------------- leakage ---
def test_latent_processing_time_is_not_exposed(frame):
    """The outcome is derived from a latent duration that must not be a column."""
    forbidden = {"actual_tat", "actual_tat_minutes", "delay_minutes", "latent"}

    assert forbidden.isdisjoint(frame.columns)


def test_no_single_feature_almost_determines_the_target(frame):
    """A near-perfect univariate correlation would indicate leakage."""
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=[TARGET_COLUMN])

    correlations = numeric.corrwith(frame[TARGET_COLUMN]).abs()
    assert correlations.max() < 0.9


# ---------------------------------------------------------------- signal ---
def test_delay_rises_with_queue_depth(frame):
    """The queue is the generator's primary pressure term."""
    quartiles = frame.groupby(
        pd.qcut(frame["queue_size"], 4, duplicates="drop"), observed=True
    )[TARGET_COLUMN].mean()

    assert quartiles.iloc[-1] > quartiles.iloc[0] * 1.5


def test_degraded_analysers_delay_more(frame):
    by_state = frame.groupby("analyser_degraded")[TARGET_COLUMN].mean()

    assert by_state.loc[1] > by_state.loc[0]


def test_under_resourced_branches_delay_more(frame):
    by_branch = frame.groupby("branch")[TARGET_COLUMN].mean()
    worst = by_branch.idxmax()
    best = by_branch.idxmin()

    assert BRANCH_PROFILES[worst].capacity < BRANCH_PROFILES[best].capacity


def test_urgent_requests_are_expedited(frame):
    by_priority = frame.groupby("priority")[TARGET_COLUMN].mean()

    assert by_priority["urgent"] < by_priority["routine"]


# -------------------------------------------------------------- metadata ---
def test_describe_marks_the_data_as_synthetic(frame):
    summary = describe_dataset(frame)

    assert summary["synthetic"] is True
    assert summary["dataset_version"] == DATASET_VERSION
    assert summary["rows"] == len(frame)
