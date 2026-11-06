"""Print per-replicate effective df so the variants can be told apart."""

from __future__ import annotations

import numpy as np
from scipy import stats

import pimeta as ma
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate
from tools.probe_edf_coverage import moments, wishart_df

for cell in ("multivariate/k05-rho30", "multivariate/k10-rho30"):
    scenario = get_scenario(cell)
    rng = np.random.default_rng(777)
    print(f"\n{cell}")
    for rep in range(6):
        data = simulate(scenario, rng)
        fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                        random="~ outcome | study", struct="UN",
                        intercept=False)
        region = ma.predict_region(fit, test="hk")
        centre = np.asarray(region.center, dtype=float)
        cov = np.asarray(region.cov, dtype=float)
        delta = np.asarray(data.theta_new, dtype=float) - centre
        d2 = float(delta @ np.linalg.solve(cov, delta))
        rows = moments(fit)
        nu_all = wishart_df(rows, "lsq")
        nu_diag = wishart_df(rows, "diag")
        J = centre.size
        def crit(nu):
            return float(J * stats.f.ppf(0.95, J, max(nu - J + 1.0, 1.0)))
        print(f"  rep {rep}: d2 {d2:7.2f}  hk {region.crit:7.2f}  "
              f"nu_all {nu_all:6.2f} -> {crit(nu_all):7.2f}   "
              f"nu_diag {nu_diag:6.2f} -> {crit(nu_diag):7.2f}   "
              f"rows {[round(r[1] / r[2], 2) for r in rows]}")
