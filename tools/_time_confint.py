"""Time `confint_tau2`, the ingredient of the upper-bound prediction interval."""

from __future__ import annotations

import time

import numpy as np

import pimeta as ma
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate

scenario = get_scenario("univariate-pi/k05-tau004")
rng = np.random.default_rng(3)
data = simulate(scenario, rng)
fit = ma.rma(list(data.yi), vi=list(data.vi), method="REML")
print("tau2_hat", float(fit.tau2))

t0 = time.perf_counter()
ci = ma.confint_tau2(list(data.yi), list(data.vi), float(fit.tau2), method="REML",
                     level=0.10)
t1 = time.perf_counter()
print(f"confint_tau2: {1000 * (t1 - t0):.1f} ms ->", ci)

t0 = time.perf_counter()
for _ in range(20):
    ma.confint_tau2(list(data.yi), list(data.vi), float(fit.tau2), method="REML",
                    level=0.10)
t1 = time.perf_counter()
print(f"mean per call: {1000 * (t1 - t0) / 20:.1f} ms -> "
      f"2000 reps x 10 cells = {2000 * 10 * (t1 - t0) / 20:.0f} s serial")

# what the upper bound does when tau2_hat is on the boundary
zeros = 0
levels = []
for rep in range(200):
    d = simulate(scenario, rng)
    f = ma.rma(list(d.yi), vi=list(d.vi), method="REML")
    if float(f.tau2) <= 1e-12:
        zeros += 1
        c = ma.confint_tau2(list(d.yi), list(d.vi), float(f.tau2), level=0.10)
        levels.append(float(c.tau2_ub))
print(f"boundary reps {zeros}/200; upper bound at the boundary: "
      f"median {np.median(levels) if levels else float('nan'):.4f}, "
      f"max {max(levels) if levels else float('nan'):.4f}")
