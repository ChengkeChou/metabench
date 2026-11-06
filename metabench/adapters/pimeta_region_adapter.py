"""`pimeta`'s joint prediction region, one method per adapter instance.

The region is an ellipsoid `(x - center)' cov^-1 (x - center) <= crit`, which
is recorded as three numbers (or a matrix) so that **containment is decided by
the metrics**, not by the adapter: an adapter that reported "inside" would be
grading its own homework.

``test="bootstrap"`` re-simulates and refits the model, so it costs ``n_boot``
fits per replicate instead of one.  The simulator is built here from the fitted
model: each replicate draws ``theta_i ~ N(0, Sigma_hat)`` per study, adds the
fitted fixed effects, draws sampling error with the *known* within-study
covariance, and refits.  That is a parametric bootstrap of the fitted model --
the reference carries the estimation error of ``Sigma_hat`` because each
bootstrap replicate builds its own region from its own ``Sigma_hat``.

Reproducibility: the seed is ``seed + replicate counter``, so a run with the
same ``--seed`` reproduces the critical values exactly even though the bootstrap
itself is random.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

import pimeta as ma

from .base import FitRecord

TESTS = {"chi2": "z", "f": "t", "hk": "hk", "bootstrap": "bootstrap"}


def _make_simulator(fit, data):
    """A parametric bootstrap step: draw from the fitted model, refit, return.

    The draw is ``y = X beta_hat + theta + eps`` with ``theta_i ~ N(0, G_hat)``
    drawn per study and ``eps ~ N(0, V)`` with ``V`` the within-study covariance
    the original data carried.  ``V`` is treated as known, exactly as the fit
    treats it.
    """
    beta = np.asarray(fit.beta, dtype=float).ravel()
    G, _names = fit.g_matrix()
    if G is None:
        raise ValueError("this fit has no between-study covariance matrix")
    sigma = np.asarray(G, dtype=float)
    X = np.asarray(data.X, dtype=float)
    V = np.asarray(data.V, dtype=float)
    rows = X.shape[0]
    J = sigma.shape[0]
    studies = list(dict.fromkeys(list(data.frame["study"])))
    position = {name: i for i, name in enumerate(studies)}
    study_index = np.array([position[name] for name in data.frame["study"]])
    outcomes = list(data.frame["outcome"])
    outcome_index = np.array([outcomes.index(name) for name in outcomes])

    def simulate(rng: np.random.Generator):
        theta = rng.multivariate_normal(np.zeros(J), sigma, size=len(studies))
        noise = rng.multivariate_normal(np.zeros(rows), V)
        y = X @ beta + theta[study_index, outcome_index] + noise
        return ma.rma_mv(y, V=V, mods=X, data=data.frame,
                         random="~ outcome | study", struct="UN",
                         intercept=False)

    return simulate


class PimetaRegionAdapter:
    name = "pimeta-region"
    version = getattr(ma, "__version__", "unknown")

    def __init__(self, method: str = "f", *, n_boot: int | None = None,
                 seed: int = 20261105, scale: float = 1.0) -> None:
        if method not in TESTS:
            raise ValueError(f"unknown region test {method!r}; "
                             f"known: {sorted(TESTS)}")
        self.method = method
        self.scale = float(scale)
        suffix = "" if self.scale == 1.0 else f"@{self.scale:g}"
        self.name = f"pimeta-region/{method}{suffix}"
        env_boot = os.environ.get("METABENCH_NBOOT")
        self.n_boot = int(n_boot or env_boot or 200)
        self.seed = int(seed)
        self._calls = 0

    def fit(self, data, *, level: float) -> FitRecord:
        record = FitRecord()
        self._calls += 1
        kwargs: dict = {}
        if self.method == "bootstrap":
            kwargs = {"n_boot": self.n_boot, "seed": self.seed + self._calls}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                                random="~ outcome | study", struct="UN",
                                intercept=False)
                if self.method == "bootstrap":
                    kwargs["simulate"] = _make_simulator(fit, data)
                region = ma.predict_region(fit, level=level,
                                           test=TESTS[self.method], **kwargs)
                if self.scale != 1.0:
                    region.crit = float(region.crit) * self.scale
                # the marginal intervals the region implies use the region's
                # own critical value, so they stay conservative projections
                gamma = float(np.sqrt(region.crit))
            record.warnings = len(caught)
        except Exception as exc:                              # noqa: BLE001
            record.failure = f"{type(exc).__name__}: {exc}"
            return record

        center = np.asarray(region.center, dtype=float)
        cov = np.asarray(region.cov, dtype=float)
        sd = np.sqrt(np.diag(cov))
        record.region_center = [float(x) for x in center]
        record.region_cov = [[float(x) for x in row] for row in cov]
        record.region_crit = float(region.crit)
        record.marginal_lb = [float(c - gamma * s) for c, s in zip(center, sd)]
        record.marginal_ub = [float(c + gamma * s) for c, s in zip(center, sd)]
        record.estimate = float(center[0])
        record.se = float(sd[0])
        record.ci_lb = record.marginal_lb[0]
        record.ci_ub = record.marginal_ub[0]
        record.tau2 = float(np.asarray(fit.tau2, dtype=float)[0])
        record.converged = bool(getattr(fit, "converged", True))
        record.boundary_hit = bool(np.any(np.asarray(fit.tau2, dtype=float)
                                          <= 0.0))
        record.valid_covariance = bool(np.all(np.isfinite(cov))
                                       and np.linalg.eigvalsh(cov).min() > 0)
        record.extra["n_boot"] = self.n_boot if self.method == "bootstrap" else 0
        return record
