"""`pimeta`'s joint prediction region, one method per adapter instance.

The region is an ellipsoid `(x - center)' cov^-1 (x - center) <= crit`, which
is recorded as three numbers (or a matrix) so that **containment is decided by
the metrics**, not by the adapter: an adapter that reported "inside" would be
grading its own homework.

``test="bootstrap"`` is not wired up here: it re-simulates and refits the model,
so it is hundreds of fits per replicate rather than one, and it belongs in its
own run with its own rep count.
"""

from __future__ import annotations

import warnings

import numpy as np

import pimeta as ma

from .base import FitRecord

TESTS = {"chi2": "z", "f": "t", "hk": "hk"}


class PimetaRegionAdapter:
    name = "pimeta-region"
    version = getattr(ma, "__version__", "unknown")

    def __init__(self, method: str = "f") -> None:
        if method not in TESTS:
            raise ValueError(f"unknown region test {method!r}; "
                             f"known: {sorted(TESTS)}")
        self.method = method
        self.name = f"pimeta-region/{method}"

    def fit(self, data, *, level: float) -> FitRecord:
        record = FitRecord()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                                random="~ outcome | study", struct="UN",
                                intercept=False)
                region = ma.predict_region(fit, level=level,
                                           test=TESTS[self.method])
                # a separate fit with the Wald reference, for the marginal
                # intervals the region implies
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
        return record
