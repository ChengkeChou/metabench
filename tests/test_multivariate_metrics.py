"""Containment is decided here, not by the adapter that produced the region.

An adapter that answered "inside" would be grading its own homework; these
tests fix the arithmetic that the harness applies to every region.
"""

from __future__ import annotations

import math

from metabench.metrics import multivariate


def record(center=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0)), crit=5.99,
           truth=(0.5, 0.5), failure=None):
    return {
        "region_center": list(center),
        "region_cov": [list(row) for row in cov],
        "region_crit": crit,
        "marginal_lb": [center[0] - math.sqrt(cov[0][0] * crit),
                        center[1] - math.sqrt(cov[1][1] * crit)],
        "marginal_ub": [center[0] + math.sqrt(cov[0][0] * crit),
                        center[1] + math.sqrt(cov[1][1] * crit)],
        "truth_new_vec": list(truth),
        "truth_vec": [0.0, 0.0],
        "boundary_hit": False,
        "converged": True,
        "failure": failure,
    }


def test_a_point_inside_the_ellipsoid_is_covered_and_one_outside_is_not():
    inside = record(truth=(1.0, 1.0))       # 2.0 <= 5.99
    outside = record(truth=(3.0, 3.0))      # 18.0 > 5.99
    result = multivariate.joint_coverage([inside, outside], key="truth_new_vec")
    assert result["hits"] == 1
    assert result["estimate"] == 0.5
    assert result["reps_used"] == 2


def test_an_elongated_region_uses_the_covariance_not_the_margins():
    # a point on the *minor* axis, outside the ellipse but inside both marginal
    # intervals: the joint region is tighter than the product of its margins
    cov = ((4.0, 3.0), (3.0, 4.0))         # eigenvalues 7 and 1
    point = (math.sqrt(2.0), -math.sqrt(2.0))
    rec = record(cov=cov, crit=2.0, truth=point)
    result = multivariate.joint_coverage([rec], key="truth_new_vec")
    assert result["estimate"] == 0.0        # quadform 4.0 > 2.0
    marginals = multivariate.marginal_coverage([rec], key="truth_new_vec")
    assert marginals[0]["estimate"] == 1.0  # |x| = 1.41 < sqrt(4 * 2) = 2.83
    assert marginals[1]["estimate"] == 1.0


def test_failed_replicates_are_counted_in_the_denominator():
    records = [record(), record(failure="ValueError: boom")]
    result = multivariate.joint_coverage(records, key="truth_new_vec")
    assert result["reps_used"] == 1
    assert result["reps_total"] == 2
    # one hit in two replicates once the failure is kept in the denominator
    assert result["with_failures"] == 0.5
    assert result["estimate"] == 1.0


def test_the_volume_proxy_is_monotone_in_the_covariance():
    small = record(cov=((1.0, 0.0), (0.0, 1.0)))
    large = record(cov=((4.0, 0.0), (0.0, 4.0)))
    assert multivariate.volume([small]) == 5.99
    assert multivariate.volume([large]) > multivariate.volume([small])


def test_summarise_reports_joint_mean_and_new_separately():
    records = [record(truth=(1.0, 1.0))]
    summary = multivariate.summarise(records)
    assert summary["joint_new"]["estimate"] == 1.0
    assert summary["joint_mean"]["estimate"] == 1.0
    assert summary["reps"] == 1
