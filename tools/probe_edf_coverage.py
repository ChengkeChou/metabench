"""Measure an effective-df reference against the closed forms, on one seed.

For every replicate: fit, compute the observed information of the variance
parameters, propagate it to ``vech(Sigma)``, match a scaled Wishart to get
``nu_eff``, and refer the pivot to ``J F(J, nu_eff - J + 1)``.  Containment is
decided from the recorded ellipsoid, exactly as the metrics do it.

Three variants are compared, because the raw moment match collapses at ``k = 5``
where the information matrix is nearly singular in the correlation direction:

``lsq``
    one ``nu`` for all elements, by least squares over the moment equations;
``diag``
    the same, restricted to the variances (which are better identified);
``floor``
    ``lsq`` with ``nu >= J + 1``, so the reference cannot degenerate to
    ``F(J, 1)``.

This is a prototype: it reaches into the package for the objective, which an
adapter must not do.  If it works, the pieces move into ``pimeta`` and the
measurement is repeated through the public API.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

import pimeta as ma
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate
from tools._probe_effective_df import (numerical_hessian, objective,
                                       sigma_and_gradient)


def moments(fit):
    """``[(is_diagonal, numerator, variance), ...]`` of the Wishart match."""
    params = np.asarray(fit.params, dtype=float)
    H = numerical_hessian(lambda p: objective(fit, p), params)
    try:
        cov_params = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    sigma, _vec0, jac, idx = sigma_and_gradient(fit, params)
    cov_vec = jac @ cov_params @ jac.T
    rows = []
    for a, (i, j) in enumerate(idx):
        var = float(cov_vec[a, a])
        if var > 0 and np.isfinite(var):
            rows.append((i == j, sigma[i, j] ** 2 + sigma[i, i] * sigma[j, j],
                         var))
    return rows or None


def wishart_df(rows, how: str = "lsq") -> float | None:
    if not rows:
        return None
    selected = [r for r in rows if (r[0] or how != "diag")]
    if not selected:
        return None
    num = np.array([r[1] for r in selected])
    den = np.array([r[2] for r in selected])
    return float(num @ num / (num @ den))


def main() -> int:
    print(f"{'cell':<14} {'reps':>5} {'hk':>7} {'lsq':>7} {'diag':>7} "
          f"{'floor':>7} {'nu_med':>7}")
    for cell in ("multivariate/k05-rho30", "multivariate/k10-rho30",
                 "multivariate/k20-rho30", "multivariate/k10-rho70"):
        scenario = get_scenario(cell)
        rng = np.random.default_rng(777)
        hits = {key: [] for key in ("hk", "lsq", "diag", "floor")}
        nus = []
        for _ in range(120):
            data = simulate(scenario, rng)
            fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                            random="~ outcome | study", struct="UN",
                            intercept=False)
            region = ma.predict_region(fit, test="hk")
            centre = np.asarray(region.center, dtype=float)
            cov = np.asarray(region.cov, dtype=float)
            delta = np.asarray(data.theta_new, dtype=float) - centre
            try:
                d2 = float(delta @ np.linalg.solve(cov, delta))
            except np.linalg.LinAlgError:
                continue
            J = centre.size
            hits["hk"].append(d2 <= region.crit)
            rows = moments(fit)
            for key in ("lsq", "diag", "floor"):
                if rows is None:
                    hits[key].append(False)
                    continue
                nu = wishart_df(rows, "diag" if key == "diag" else "lsq")
                if nu is None:
                    hits[key].append(False)
                    continue
                if key == "floor":
                    nu = max(nu, J + 1.0)
                    nus.append(nu)
                df2 = max(nu - J + 1.0, 1.0)
                crit = float(J * stats.f.ppf(0.95, J, df2))
                hits[key].append(d2 <= crit)
        if not hits["hk"]:
            continue
        print(f"{cell:<14} {len(hits['hk']):>5} {np.mean(hits['hk']):>7.3f} "
              f"{np.mean(hits['lsq']):>7.3f} {np.mean(hits['diag']):>7.3f} "
              f"{np.mean(hits['floor']):>7.3f} "
              f"{np.median(nus) if nus else float('nan'):>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
