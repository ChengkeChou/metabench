"""The metrics must be recomputable from raw records, and must count failures.

The two failure modes this file guards against are the ones that make a
simulation study flattering rather than informative: dropping replicates that
did not fit, and reporting a proportion without its Monte Carlo error.
"""

from __future__ import annotations

import math

from metabench.metrics import coverage, mcse_proportion, rates, summarise


def record(**overrides):
    base = {
        "estimate": 0.0, "se": 0.1,
        "ci_lb": -0.2, "ci_ub": 0.2,
        "pi_lb": -0.3, "pi_ub": 0.3,
        "pi_t_lb": -0.3, "pi_t_ub": 0.3,
        "truth": 0.0, "truth_new": 0.0,
        "boundary_hit": False, "converged": True,
        "valid_covariance": True, "failure": None,
    }
    base.update(overrides)
    return base


def test_a_failed_replicate_is_counted_and_both_coverages_are_reported():
    records = [
        record(),
        record(),
        record(ci_lb=1.0, ci_ub=2.0),              # a miss
        record(failure="ValueError: nope", ci_lb=None, ci_ub=None),
    ]
    result = coverage(records, lo="ci_lb", hi="ci_ub", truth="truth")
    assert result["hits"] == 2
    assert result["reps_used"] == 3             # the failure is excluded
    assert result["reps_total"] == 4            # but it is still counted
    assert result["estimate"] == 2 / 3
    assert result["with_failures"] == 2 / 4     # failures count as misses
    assert rates(records)["failure_rate"] == 0.25


def test_every_proportion_carries_its_monte_carlo_error():
    records = [record() for _ in range(60)]
    result = coverage(records, lo="ci_lb", hi="ci_ub", truth="truth")
    assert result["reps_used"] == 60
    expected = math.sqrt(1.0 * 0.0 / 60)        # all hits -> zero SE here
    assert result["mcse"] == expected
    assert result["halfwidth_95"] == 0.0


def test_summarise_reports_the_rates_the_estimand_and_the_widths():
    records = [record(), record(boundary_hit=True, converged=False)]
    summary = summarise(records)
    assert summary["reps"] == 2
    assert summary["ci_width"] == 0.4
    assert summary["pi_width"] == 0.6
    assert summary["rates"]["boundary_rate"] == 0.5
    assert summary["rates"]["non_converged_rate"] == 0.5
    assert summary["bias"]["estimate"] == 0.0


def test_mcse_shrinks_like_a_root_n():
    assert abs(mcse_proportion(0.95, 2000) - 0.004873) < 1e-6
    assert mcse_proportion(0.95, 60) > 4 * mcse_proportion(0.95, 2000)
