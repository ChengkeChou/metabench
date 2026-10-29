"""`pimeta` as one implementation among several, through its public API.

Only ``pimeta.rma`` and the result's documented attributes are used.  Both
prediction intervals are recorded -- the default (which mirrors `metafor`) and
``predtype="t"`` (which charges the t distribution for the uncertainty in
tau^2) -- because they answer the same question with different coverage and the
harness exists to keep those apart.
"""

from __future__ import annotations

import warnings

import numpy as np

import pimeta as ma

from .base import FitRecord


class PimetaAdapter:
    name = "pimeta"
    version = getattr(ma, "__version__", "unknown")

    def __init__(self, method: str = "REML") -> None:
        self.method = method

    def fit(self, data, *, level: float) -> FitRecord:
        record = FitRecord()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = ma.rma(list(data.yi), vi=list(data.vi),
                             method=self.method)
            record.warnings = len(caught)
        except Exception as exc:                              # noqa: BLE001
            record.failure = f"{type(exc).__name__}: {exc}"
            return record

        record.estimate = float(fit.b[0])
        record.se = float(fit.se[0])
        record.ci_lb = float(fit.ci_lb[0])
        record.ci_ub = float(fit.ci_ub[0])
        record.tau2 = float(fit.tau2)
        record.converged = bool(getattr(fit, "converged", True))
        record.boundary_hit = bool(record.tau2 <= 0.0)
        record.valid_covariance = bool(np.isfinite(record.se)
                                       and record.se > 0.0)

        for prefix, kwargs in (("pi", {"predtype": "default"}),
                               ("pi_t", {"predtype": "t"})):
            try:
                prediction = fit.predict(level=level, **kwargs)
                # ``predict`` returns scalars for the fitted model and arrays
                # when ``newmods`` is given; the harness always asks for the
                # fitted model, so accept either shape.
                setattr(record, f"{prefix}_lb",
                        float(np.atleast_1d(prediction.pi_lb)[0]))
                setattr(record, f"{prefix}_ub",
                        float(np.atleast_1d(prediction.pi_ub)[0]))
            except Exception as exc:                          # noqa: BLE001
                if record.failure is None:
                    record.failure = f"predict {prefix}: {type(exc).__name__}: {exc}"
        return record
