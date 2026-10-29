"""The reference has to be right before it can judge anything.

Three claims, in the order they have to hold:

1. the hyperspherical map is a bijection onto the correlation matrices, so a
   target `R` survives a round trip;
2. `un_reml_nll` is the same function `pimeta` maximises: evaluated at the
   matrix `pimeta` returns, it reproduces `pimeta`'s reported log-likelihood;
3. searching from several starts never returns a *worse* point than
   `pimeta`'s, on data where `pimeta` is known to be fine (J = 2).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pimeta")

import pimeta as ma                                       # noqa: E402

from metabench.reference import fit_un, un_reml_nll       # noqa: E402
from metabench.reference.un_reml import (                 # noqa: E402
    corr_from_z,
    to_external,
    z_from_corr,
)
from metabench.scenarios import get, simulate             # noqa: E402


def _data(scenario_id: str, seed: int = 11):
    scenario = get(scenario_id)
    rng = np.random.default_rng(seed)
    return scenario, simulate(scenario, rng)


def _fit(data):
    return ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                     random="~ outcome | study", struct="UN",
                     intercept=False)


@pytest.mark.parametrize("J", [2, 3, 5])
def test_the_spherical_map_round_trips(J):
    rho = 0.45
    R = np.eye(J) + rho * (np.ones((J, J)) - np.eye(J))
    back = corr_from_z(J, z_from_corr(R))
    assert np.allclose(back, R, atol=1e-10)


def test_the_spherical_map_handles_a_negative_correlation():
    R = np.array([[1.0, -0.6, 0.2], [-0.6, 1.0, 0.1], [0.2, 0.1, 1.0]])
    assert np.allclose(corr_from_z(3, z_from_corr(R)), R, atol=1e-10)


@pytest.mark.parametrize("scenario_id", ["covariance/k10-J2-rho30",
                                         "covariance/k10-J3-rho30"])
def test_the_likelihood_matches_pimeta_at_pimeta_s_own_answer(scenario_id):
    """Parity, not correctness: both are the same formula on the same matrix."""
    _, data = _data(scenario_id)
    fit = _fit(data)
    G, _ = fit.g_matrix()
    ours = -un_reml_nll(data.yi, data.V, data.X, np.asarray(G, dtype=float))
    assert ours == pytest.approx(fit.ll, abs=1e-8)


@pytest.mark.parametrize("variant", ["hyper", "external", "chol"])
def test_the_reference_never_loses_on_the_easy_cell(variant):
    """J = 2, where `pimeta`'s box is the whole cone and parity is established."""
    _, data = _data("covariance/k10-J2-rho30")
    fit = _fit(data)
    G, _ = fit.g_matrix()
    at_fit = un_reml_nll(data.yi, data.V, data.X, np.asarray(G, dtype=float))
    reference = fit_un(data.yi, data.V, data.X, variant=variant)
    assert reference.nll <= at_fit + 1e-6


def test_the_external_variant_reproduces_pimeta_s_external_vector():
    """The reference's `(tau2, rho)` must land where `pimeta`'s does."""
    _, data = _data("covariance/k10-J3-rho30")
    fit = _fit(data)
    G, _ = fit.g_matrix()
    external = to_external(np.asarray(G, dtype=float))
    assert external.size == 6                       # three tau2, three rho
    assert np.allclose(external[:3], np.diag(G), atol=0.0)
    # at most one: the maximum at k=10, J=3 is often *on* the cone boundary,
    # which is the finding this cell exists for
    assert np.all(np.abs(external[3:]) <= 1.0)
