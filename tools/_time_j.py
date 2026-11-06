"""Time `rma_mv` for J = 2, 3, 5 so the next grid can be sized before it runs."""

from __future__ import annotations

import time

import numpy as np

import pimeta as ma
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate
from metabench.scenarios.multivariate import scenario

CELLS = {"J2": get_scenario("multivariate/k10-rho30")}
for J in (3, 5):
    cell = scenario(f"timing/J{J}", k=10, tau2=tuple([0.04 + 0.01 * j for j in range(J)]),
                    se=tuple([0.2 + 0.01 * j for j in range(J)]), rho=0.3,
                    mu=tuple([0.0] * J))
    CELLS[f"J{J}"] = cell

for name, cell in CELLS.items():
    data = simulate(cell, np.random.default_rng(5))
    ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
              random="~ outcome | study", struct="UN", intercept=False)
    t0 = time.perf_counter()
    for _ in range(5):
        fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                        random="~ outcome | study", struct="UN",
                        intercept=False)
    dt = (time.perf_counter() - t0) / 5
    print(f"{name}: {1000 * dt:.0f} ms per fit, k = {data.yi.size // len(cell.dgm['mu'])}")
