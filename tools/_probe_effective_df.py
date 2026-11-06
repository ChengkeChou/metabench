"""Can an effective-df correction be *derived* from the fit's own information?

The measured multiplier on the F/HK critical value is 2.1-4.4x and is not
monotone in k, which says the problem is the number of *effective* degrees of
freedom in ``Sigma-hat``, not a constant scale.  This probe computes the observed
information of the variance parameters numerically, propagates it to
``vech(Sigma)`` by the delta method, matches the result to a scaled Wishart to get
``nu_eff``, and compares the implied critical value with the one measured to be
needed.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats

import pimeta as ma
from pimeta.rma_mv import _independent_blocks, _mv_loglik
from metabench.scenarios import get as get_scenario
from metabench.scenarios import simulate


def objective(fit, params):
    """The fit's objective at a parameter vector, rebuilt from the result.

    ``_mv_loglik`` returns **minus** the log-likelihood (the optimiser minimises
    it), so this is already the quantity whose Hessian is the observed
    information.
    """
    D_list = []
    offset = 0
    for comp in fit.components:
        piece = params[offset:offset + comp.n_params]
        offset += comp.n_params
        D_list.append(comp.make_D(piece))
    groups = _independent_blocks(fit.components, fit.V, int(fit.yi.size),
                                 D_list[0])
    return _mv_loglik(D_list, np.asarray(fit.yi, dtype=float).reshape(-1, 1),
                      np.asarray(fit.V, dtype=float),
                      np.asarray(fit.X, dtype=float), fit.method, groups=groups)


def numerical_hessian(func, x, step_scale=1e-4):
    n = x.size
    steps = step_scale * np.maximum(np.abs(x), 1e-3)
    H = np.zeros((n, n))
    f0 = func(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        H[i, i] = (func(x + ei) - 2 * f0 + func(x - ei)) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = steps[j]
            H[i, j] = H[j, i] = (
                func(x + ei + ej) - func(x + ei - ej)
                - func(x - ei + ej) + func(x - ei - ej)
            ) / (4 * steps[i] * steps[j])
    return H


def sigma_and_gradient(fit, params, h=1e-6):
    """``vech(Sigma)`` and its Jacobian with respect to the parameters.

    ``Sigma`` here is the ``J x J`` between-study covariance, taken from the
    component's ``_G`` exactly as :meth:`RmaMvResult.g_matrix` does.  Using
    ``make_D`` instead returns the *study-level block* matrix, and vech over
    that repeats each element once per study -- which is how the first version
    of this probe produced ten rows for a two-outcome model.
    """
    def sigma_at(p):
        offset = 0
        for comp in fit.components:
            piece = p[offset:offset + comp.n_params]
            offset += comp.n_params
            J_comp = int(getattr(comp, "J", getattr(comp, "q", 1)))
            if J_comp < 2:
                continue
            if hasattr(comp, "_G"):
                return np.asarray(comp._G(piece), dtype=float)
            _s2, tau2, rho = comp.report(piece)
            tau2_val = float(tau2[0]) if tau2 else 0.0
            rho_val = float(rho[0]) if rho else 0.0
            G = np.full((J_comp, J_comp), rho_val * tau2_val)
            np.fill_diagonal(G, tau2_val)
            return G
        raise ValueError("no multivariate random term")

    base = sigma_at(params)
    J = base.shape[0]
    idx = [(i, j) for i in range(J) for j in range(i, J)]
    vec0 = np.array([base[i, j] for i, j in idx])
    jac = np.zeros((len(idx), params.size))
    for k in range(params.size):
        shifted = params.copy()
        shifted[k] += h
        moved = sigma_at(shifted)
        jac[:, k] = (np.array([moved[i, j] for i, j in idx]) - vec0) / h
    return base, vec0, jac, idx


def effective_df(fit):
    params = np.asarray(fit.params, dtype=float)
    t0 = time.perf_counter()
    H = numerical_hessian(lambda p: objective(fit, p), params)
    seconds = time.perf_counter() - t0
    try:
        cov_params = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, seconds
    sigma, vec0, jac, idx = sigma_and_gradient(fit, params)
    cov_vec = jac @ cov_params @ jac.T
    values = []
    for a, (i, j) in enumerate(idx):
        var = float(cov_vec[a, a])
        if var <= 0 or not np.isfinite(var):
            continue
        values.append((sigma[i, j] ** 2 + sigma[i, i] * sigma[j, j]) / var)
    return (float(np.median(values)) if values else None), seconds


for cell in ("multivariate/k10-rho30", "multivariate/k20-rho30",
             "multivariate/k05-rho30"):
    scenario = get_scenario(cell)
    data = simulate(scenario, np.random.default_rng(21))
    fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                    random="~ outcome | study", struct="UN", intercept=False)
    nu_hat, seconds = effective_df(fit)
    J = int(np.asarray(fit.g_matrix()[0]).shape[0])
    k = int(np.unique(np.asarray(data.frame["study"])).size)
    f_ref = float(J * stats.f.ppf(0.95, J, k - J))
    hk = ma.predict_region(fit, test="hk").crit
    if nu_hat is None:
        print(f"{cell}: information singular")
        continue
    df2 = max(nu_hat - J + 1.0, 1.0)
    crit_edf = float(J * stats.f.ppf(0.95, J, df2))
    print(f"{cell}: k={k} J={J}  nu_eff={nu_hat:6.2f}  df2={df2:6.2f}  "
          f"crit(edf)={crit_edf:6.2f}  crit(F)={f_ref:6.2f}  hk={hk:6.2f}  "
          f"Hessian {1000 * seconds:.0f} ms")
