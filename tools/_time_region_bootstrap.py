"""Time the pieces the two correction studies depend on.

The joint-region bootstrap costs ``n_boot`` refits per replicate, so the run has
to be sized before it is launched rather than after it fails at 90 %.
"""

from __future__ import annotations

import time

import numpy as np

import pimeta as ma
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate

scenario = get_scenario("multivariate/k10-rho30")
rng = np.random.default_rng(7)
data = simulate(scenario, rng)
print("k =", data.yi.shape, "J =", data.V.shape[-1])

t0 = time.perf_counter()
fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                random="~ outcome | study", struct="UN", intercept=False)
t1 = time.perf_counter()
print(f"rma_mv fit: {1000 * (t1 - t0):.1f} ms")

region = ma.predict_region(fit, test="t")
print("region crit (t):", round(float(region.crit), 4))
print("beta:", np.asarray(fit.beta).ravel())
G, names = fit.g_matrix()
print("G:", np.asarray(G), names)


def simulate_rep(rng_):
    """Draw a new data set from the fitted model and refit, as the API wants."""
    beta = np.asarray(fit.beta, dtype=float).ravel()
    sigma = np.asarray(G, dtype=float)
    k = data.yi.shape[0]
    studies = list(dict.fromkeys(list(data.frame["study"])))
    theta = rng_.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=len(studies))
    yi = np.empty_like(data.yi, dtype=float)
    outcome = list(data.frame["outcome"])
    index = {name: i for i, name in enumerate(studies)}
    for row in range(k):
        study_index = index[data.frame["study"][row]]
        outcome_index = outcome.index(outcome[row])
        # X is the intercept-only design in this scenario: mu = beta
        mean = beta[outcome_index] + theta[study_index, outcome_index]
        yi[row] = rng_.normal(mean, np.sqrt(data.V[row, row]))
    return ma.rma_mv(yi, V=data.V, mods=data.X, data=data.frame,
                     random="~ outcome | study", struct="UN", intercept=False)


t0 = time.perf_counter()
for _ in range(20):
    simulate_rep(np.random.default_rng(1))
t1 = time.perf_counter()
print(f"refit: {1000 * (t1 - t0) / 20:.1f} ms -> "
      f"200 boots = {200 * (t1 - t0) / 20:.2f} s per replicate")

t0 = time.perf_counter()
region_boot = ma.predict_region(fit, test="bootstrap", simulate=simulate_rep,
                                n_boot=40, seed=11)
t1 = time.perf_counter()
print(f"bootstrap(40): {t1 - t0:.2f} s, crit = {float(region_boot.crit):.4f}")
print(f"extrapolated 200 boots: {5 * (t1 - t0):.2f} s per replicate")
