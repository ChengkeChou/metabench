"""The covariance metrics, checked on records whose answers are known by hand."""

from __future__ import annotations

import numpy as np
import pytest

from metabench.metrics import covariance as cm
from metabench.reports.table import row_for
from metabench.scenarios import get, simulate


def _record(sigma, truth, *, gap=0.0, boundary=False, valid=True,
            n_at_best=6, runtime=0.1):
    return {
        "sigma_hat": [[float(x) for x in row] for row in np.asarray(sigma)],
        "truth_cov": [[float(x) for x in row] for row in np.asarray(truth)],
        "boundary_hit": boundary,
        "valid_covariance": valid,
        "runtime": runtime,
        "extra": {
            "gap": gap,
            "ref_spread": 0.0,
            "ref_n_at_best": n_at_best,
            "symmetric": True,
            "mapping_gap": 0.0,
            "ll_library": -1.0,
            "ll_at_fit": -1.0,
        },
    }


TRUTH = np.array([[0.04, 0.018], [0.018, 0.09]])


def test_a_perfect_fit_with_zero_gap_is_not_suboptimal():
    summary = cm.summarise([_record(TRUTH, TRUTH)])
    assert summary["rates"]["suboptimal_rate"] == 0.0
    assert summary["frobenius"]["median"] == pytest.approx(0.0, abs=1e-12)
    assert summary["tau2_relative_bias"] == pytest.approx(0.0, abs=1e-12)
    assert summary["rho_bias"] == pytest.approx(0.0, abs=1e-12)


def test_a_fit_above_the_tolerance_is_counted_as_suboptimal():
    summary = cm.summarise([_record(TRUTH, TRUTH, gap=1.0),
                            _record(TRUTH, TRUTH, gap=0.0)])
    assert summary["rates"]["suboptimal_rate"] == 0.5
    assert summary["gap"]["max"] == pytest.approx(1.0)
    assert summary["proportions"]["suboptimal"]["mcse"] == pytest.approx(
        np.sqrt(0.25 / 2))


def test_the_reference_winning_is_counted_separately():
    summary = cm.summarise([_record(TRUTH, TRUTH, gap=-0.5)])
    assert summary["rates"]["better_rate"] == 1.0
    assert summary["rates"]["suboptimal_rate"] == 0.0


def test_a_boundary_fit_is_not_a_valid_covariance():
    summary = cm.summarise([_record(TRUTH, TRUTH, boundary=True, valid=False)])
    assert summary["rates"]["boundary_rate"] == 1.0
    assert summary["rates"]["non_pd_rate"] == 1.0


def test_failures_stay_in_the_denominator():
    failed = {"sigma_hat": None, "failure": "ValueError: boom"}
    summary = cm.summarise([_record(TRUTH, TRUTH), failed])
    assert summary["reps"] == 2
    assert summary["usable"] == 1
    assert summary["rates"]["failure_rate"] == 0.5


def test_a_start_that_never_found_the_best_point_shows_as_conflict():
    summary = cm.summarise([_record(TRUTH, TRUTH, n_at_best=1)])
    assert summary["rates"]["multistart_conflict_rate"] == 1.0


def test_the_table_row_prints_the_numbers_it_was_given():
    summary = cm.summarise([_record(TRUTH * 2, TRUTH)])
    row = row_for({
        "scenario": {"scenario_id": "covariance/synthetic", "family": "covariance"},
        "adapter": "test",
        "metrics": summary,
    })
    assert row[0] == "covariance/synthetic"
    assert row[3] == "1"
    assert row[10] != "—"                     # the relative Frobenius error


def test_the_covariance_family_records_a_positive_definite_truth():
    scenario = get("covariance/k10-J3-rho30")
    data = simulate(scenario, np.random.default_rng(3))
    assert np.all(np.linalg.eigvalsh(data.Sigma) > 0)
    assert data.Sigma.shape == (3, 3)
    assert data.X.shape == (30, 3)
    # V is block diagonal per study and X is the outcome identity per study
    assert np.allclose(data.V, np.kron(np.eye(10), np.diag([0.20, 0.25, 0.30]) ** 2))
    assert np.allclose(data.X[:3], np.eye(3))
