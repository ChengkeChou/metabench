"""Debug why the numerical information matrix came out singular."""

from __future__ import annotations

import numpy as np

import pimeta as ma
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate
from tools._probe_effective_df import negative_loglik, numerical_hessian

data = simulate(get_scenario("multivariate/k10-rho30"), np.random.default_rng(21))
fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                random="~ outcome | study", struct="UN", intercept=False)
params = np.asarray(fit.params, dtype=float)
print("params:", params)
print("components:", [(c.n_params, getattr(c, "kind", None)) for c in fit.components])
print("obj at optimum:", negative_loglik(fit, params))

for step in (1e-2, 1e-3, 1e-4, 1e-5):
    H = numerical_hessian(lambda p: negative_loglik(fit, p), params, step)
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    print(f"step {step:g}: eigenvalues {np.round(eig, 6)}  "
          f"cond {np.linalg.cond(H):.3g}")

# is the optimum really an interior minimum of the rebuilt objective?
for i in range(params.size):
    for delta in (-1e-2, 1e-2):
        moved = params.copy()
        moved[i] += delta
        value = negative_loglik(fit, moved)
        print(f"  param {i} {delta:+.2f}: {value:.8f} "
              f"({value - negative_loglik(fit, params):+.2e})")

# and what the fit itself reports
print("\nfit.logLik:", fit.logLik if hasattr(fit, "logLik") else "n/a")
print("fit.ll:", getattr(fit, "ll", None))
print("sigma2:", getattr(fit, "sigma2", None))
print("boundary:", [float(x) <= 1e-10 for x in np.asarray(fit.tau2, dtype=float).ravel()])
