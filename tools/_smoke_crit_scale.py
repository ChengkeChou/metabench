"""Smoke-test `crit_scale` on a real fit."""

from __future__ import annotations

import numpy as np

import pimeta as ma
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate

data = simulate(get_scenario("multivariate/k10-rho30"), np.random.default_rng(11))
fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                random="~ outcome | study", struct="UN", intercept=False)

plain = ma.predict_region(fit, test="hk")
scaled = ma.predict_region(fit, test="hk", crit_scale="measured")
print(f"hk crit          {plain.crit:.4f}")
print(f"hk scaled crit   {scaled.crit:.4f}  (x{scaled.crit_scale:.2f})")
print(f"contains truth?  plain {plain.contains(data.theta_new)}  "
      f"scaled {scaled.contains(data.theta_new)}")
print(scaled)

print("\nexplicit number:")
print(f"  crit x3.0 -> {ma.predict_region(fit, test='t', crit_scale=3.0).crit:.4f}")

for bad in ({"test": "t", "crit_scale": "measured"},
            {"test": "hk", "crit_scale": "nonsense"},
            {"test": "hk", "crit_scale": -1.0}):
    try:
        ma.predict_region(fit, **bad)
        print(f"  no error for {bad}  <-- unexpected")
    except ValueError as exc:
        print(f"  refused {bad}: {str(exc)[:70]}")
