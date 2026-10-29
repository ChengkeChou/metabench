"""The metric tests for the covariance family, plus one guard that was earned."""

from __future__ import annotations

import numpy as np
import pytest

from metabench.reference import UNLikelihood
from metabench.reference.un_reml import fit_un
from metabench.scenarios import get, simulate


def _data(cell: str = "covariance/k10-J3-rho30", seed: int = 20261029,
          skip: int = 0):
    scenario = get(cell)
    rng = np.random.default_rng(seed)
    data = None
    for _ in range(skip + 1):
        data = simulate(scenario, rng)
    return data


def test_an_absurdly_large_covariance_is_refused_rather_than_scored():
    """A likelihood better than any likelihood is arithmetic, not statistics.

    In a real run at k=10, J=3 one start wandered to ``tau2 ~ 1e43`` and
    reported ``nll = -2648.9`` on data whose best value is ``9.27``: at that
    scale ``log|M|`` and ``log|X'M^-1 X|`` are two numbers of order
    ``kJ log(lambda)`` that cancel *past* the minimum in floating point.  The
    reference must refuse the region instead of reporting it, because a
    reference that can be 2658 log-likelihood units better than the truth
    would have declared every honest fit suboptimal.
    """
    data = _data()
    likelihood = UNLikelihood(data.yi, data.V, data.X)
    absurd = np.eye(3) * 1e43
    assert likelihood.nll(absurd) > 1e20


def test_the_reference_optimum_is_not_better_than_a_perfect_likelihood():
    """nll is bounded below by the impossible-perfect fit, not by -infinity."""
    data = _data(skip=184)          # the replicate that exposed the bug
    reference = fit_un(data.yi, data.V, data.X, variant="hyper")
    assert reference.nll > 0.0
    assert all(value > 0.0 for value in reference.start_nlls)


@pytest.mark.parametrize("variant", ["hyper", "external", "chol"])
def test_the_three_maps_agree_where_the_optimum_is_interior(variant):
    data = _data(cell="covariance/k30-J3-rho30", seed=99)
    likelihood = UNLikelihood(data.yi, data.V, data.X)
    reference = fit_un(data.yi, data.V, data.X, variant=variant)
    at_truth = likelihood.nll(data.Sigma)
    assert reference.nll <= at_truth + 1e-6
    assert reference.nll > 0.0
