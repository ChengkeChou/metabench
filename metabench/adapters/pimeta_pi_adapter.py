"""Prediction intervals built by a named small-sample rule.

The shipped interval plugs :math:`\\hat\\tau^2` into
``mu +- t * sqrt(tau2 + se^2)``.  Measured, that interval is short wherever
``tau^2`` is estimated badly, and the diagnosis says why: its width collapses
when :math:`\\hat\\tau^2` lands on the boundary, while the distance it has to
cover does not collapse with it.  The remedy that follows from that diagnosis is
to stop letting the width collapse — build it from an **upper confidence bound**
for :math:`\\tau^2` instead of the point estimate.  That is conservative by
construction and cannot be narrower than the shipped interval.

Each rule is one adapter instance, and the shipped interval is recorded
alongside in ``extra`` so the width a rule costs is visible in the same record.

Rules:

``upper95``
    95 % one-sided upper bound for tau^2 (QP profile), t with ``k - 1`` df.
``upper80``
    the same at 90 % one-sided, the cheaper version of the same idea.
``upper95t``
    the 95 % upper bound with ``t`` on ``k - 2`` df.
``blend20``
    a fifth of the way from the point estimate to the 95 % upper bound,
    :math:`\\hat\\tau^2 + 0.2(\\tau^2_{ub} - \\hat\\tau^2)`.  The full bound is a
    large over-correction for a prediction interval (measured: 0.99 at four
    times the width), so the usable version is a partial one.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

import pimeta as ma

from .base import FitRecord

RULES = ("upper95", "upper80", "upper95t", "blend20")

#: two-sided alpha whose *upper* limit is the one-sided bound of the rule
ALPHA = {"upper95": 0.10, "upper80": 0.20, "upper95t": 0.10, "blend20": 0.10}

#: how far to move from tau2_hat towards that bound
LAMBDA = {"upper95": 1.0, "upper80": 1.0, "upper95t": 1.0, "blend20": 0.2}

#: degrees of freedom charged against k
DF_OFFSET = {"upper95": 1, "upper80": 1, "upper95t": 2, "blend20": 1}


class PimetaPiAdapter:
    name = "pimeta-pi"
    version = getattr(ma, "__version__", "unknown")

    def __init__(self, rule: str = "upper95", *, method: str = "REML") -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; known: {list(RULES)}")
        self.rule = rule
        self.method = method
        self.name = f"pimeta-pi/{rule}"

    def fit(self, data, *, level: float) -> FitRecord:
        record = FitRecord()
        y = list(data.yi)
        v = list(data.vi)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = ma.rma(y, vi=v, method=self.method)
            record.warnings = len(caught)
        except Exception as exc:                              # noqa: BLE001
            record.failure = f"{type(exc).__name__}: {exc}"
            return record

        k = int(np.size(data.yi))
        tau2 = float(fit.tau2)
        record.estimate = float(fit.b[0])
        record.se = float(fit.se[0])
        record.ci_lb = float(fit.ci_lb[0])
        record.ci_ub = float(fit.ci_ub[0])
        record.tau2 = tau2
        record.converged = bool(getattr(fit, "converged", True))
        record.boundary_hit = bool(tau2 <= 0.0)
        record.valid_covariance = bool(np.isfinite(record.se) and record.se > 0.0)

        # the shipped interval, kept for the width comparison
        try:
            shipped = fit.predict(level=level)
            record.extra["shipped_lb"] = float(np.atleast_1d(shipped.pi_lb)[0])
            record.extra["shipped_ub"] = float(np.atleast_1d(shipped.pi_ub)[0])
        except Exception:                                     # noqa: BLE001
            pass

        try:
            ci = ma.confint_tau2(y, v, tau2, method=self.method,
                                 level=ALPHA[self.rule])
            lam = LAMBDA[self.rule]
            tau2_used = tau2 + lam * (max(float(ci.tau2_ub), tau2) - tau2)
        except Exception as exc:                              # noqa: BLE001
            record.failure = f"confint_tau2: {type(exc).__name__}: {exc}"
            return record

        df = k - DF_OFFSET[self.rule]
        if df <= 0:
            record.failure = f"no degrees of freedom left (k={k})"
            return record
        q = float(stats.t.ppf(1.0 - level_to_alpha(level) / 2.0, df))
        half = q * float(np.sqrt(tau2_used + record.se ** 2))
        record.pi_lb = record.estimate - half
        record.pi_ub = record.estimate + half
        # the shipped ``predtype="t"`` interval is a different correction and is
        # recorded too, so the two can be compared on the same replicates
        try:
            alt = fit.predict(level=level, predtype="t")
            record.pi_t_lb = float(np.atleast_1d(alt.pi_lb)[0])
            record.pi_t_ub = float(np.atleast_1d(alt.pi_ub)[0])
        except Exception:                                     # noqa: BLE001
            pass
        record.extra["tau2_used"] = tau2_used
        record.extra["df"] = df
        return record


def level_to_alpha(level: float) -> float:
    """Accept 95, 0.95 or 0.05 and return the upper tail probability."""
    value = float(level)
    if value > 1.0:
        value = value / 100.0
    return value if value < 0.5 else 1.0 - value
